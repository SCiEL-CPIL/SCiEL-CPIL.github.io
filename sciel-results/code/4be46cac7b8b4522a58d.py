import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_normals(angle_deg):
        base_angle = np.radians(angle_deg + 30)
        angles = base_angle + np.radians(np.arange(0, 360, 60))
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def compute_minimal_outer_side(inner_positions, outer_center, outer_angle):
        normals = get_normals(outer_angle)
        max_proj = 0.0
        for i in range(len(inner_positions)):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            verts_rel = verts - np.array(outer_center)
            projs = verts_rel @ normals.T
            max_proj = max(max_proj, np.max(projs))
        apothem = max_proj
        side = apothem / (math.sqrt(3) / 2)
        return side

    def polygons_disjoint_sat(v1, v2):
        """
        Check if two convex polygons are disjoint using SAT.
        Returns the minimum gap (positive if disjoint, negative if overlapping).
        """
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        
        max_gap = -float('inf')
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap1 = min2 - max1
            gap2 = min1 - max2
            gap = max(gap1, gap2)
            
            if gap > max_gap:
                max_gap = gap
                
        return max_gap

    def objective(params):
        if time.time() > deadline - 2:
            return 1000.0

        # Unpack parameters
        # params: [x1, y1, a1, ..., x12, y12, a12, outer_x, outer_y, outer_angle]
        # Total 39 params
        
        inner_positions = []
        for i in range(12):
            x = params[i*3]
            y = params[i*3 + 1]
            a = params[i*3 + 2]
            inner_positions.append((x, y, a))
        
        outer_x = params[36]
        outer_y = params[37]
        outer_angle = params[38]
        
        # Compute outer side
        os = compute_minimal_outer_side(inner_positions, (outer_x, outer_y), outer_angle)
        
        # Compute collision penalty
        penalty = 0.0
        verts_list = []
        for i in range(12):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            verts_list.append(verts)
            
        for i in range(12):
            for j in range(i+1, 12):
                ci = inner_positions[i]
                cj = inner_positions[j]
                dist_sq = (ci[0]-cj[0])**2 + (ci[1]-cj[1])**2
                if dist_sq > 12.0: # 3.46^2
                    continue
                    
                gap = polygons_disjoint_sat(verts_list[i], verts_list[j])
                # gap > 0 means disjoint.
                # gap < 0 means overlapping.
                # We want to penalize overlap.
                # Use softplus for smoothness.
                k_smooth = 20.0
                arg = -gap * k_smooth
                if arg > 50:
                    penalty += arg
                else:
                    penalty += math.log(1 + math.exp(arg))
                    
        return os + penalty * 10.0

    def simulated_annealing(initial_params, max_iter=2000, T_start=1.0, T_end=0.01):
        """
        Simulated Annealing for finding a good packing.
        """
        current_params = np.array(initial_params, dtype=float)
        current_obj = objective(current_params)
        best_params = current_params.copy()
        best_obj = current_obj
        
        for i in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            T = T_start * (T_end / T_start) ** (i / max_iter)
            
            # Perturb
            new_params = current_params.copy()
            # Perturb a random subset of parameters
            num_perturb = random.randint(1, 3)
            for _ in range(num_perturb):
                idx = random.randint(0, len(new_params) - 1)
                if idx % 3 == 2:  # Angle
                    new_params[idx] += random.gauss(0, 5.0)
                else:  # Position
                    new_params[idx] += random.gauss(0, 0.5)
            
            # Keep outer center and angle mostly fixed for stability in SA
            new_params[36] = current_params[36]
            new_params[37] = current_params[37]
            new_params[38] = current_params[38]
            
            new_obj = objective(new_params)
            
            if new_obj < current_obj:
                current_params = new_params
                current_obj = new_obj
                if new_obj < best_obj:
                    best_params = new_params.copy()
                    best_obj = new_obj
            else:
                # Accept with probability exp(-delta/T)
                delta = new_obj - current_obj
                if delta < 0:
                    accept_prob = 1.0
                else:
                    accept_prob = math.exp(-delta / T)
                
                if random.random() < accept_prob:
                    current_params = new_params
                    current_obj = new_obj
        
        return best_params, best_obj

    # Generate initial guesses
    # Strategy: Start with a hexagonal lattice pattern
    def generate_initial_guess():
        params = np.zeros(39)
        
        # Center hexagon
        params[0] = 0.0
        params[1] = 0.0
        params[2] = 0.0
        
        # Ring 1: 6 hexagons
        r1 = 2.05
        for k in range(6):
            idx = 1 + k
            angle = np.radians(60 * k)
            params[idx*3] = r1 * np.cos(angle)
            params[idx*3 + 1] = r1 * np.sin(angle)
            params[idx*3 + 2] = 0.0
        
        # Ring 2: 5 hexagons (leaving one spot empty or shifting)
        r2 = 3.6
        for k in range(5):
            idx = 7 + k
            angle = np.radians(60 * k + 30)
            params[idx*3] = r2 * np.cos(angle)
            params[idx*3 + 1] = r2 * np.sin(angle)
            params[idx*3 + 2] = 30.0
            
        # Outer center and angle
        params[36] = 0.0
        params[37] = 0.0
        params[38] = 30.0
        
        return params

    def generate_random_guess():
        params = np.zeros(39)
        for i in range(12):
            params[i*3] = random.uniform(-3, 3)
            params[i*3 + 1] = random.uniform(-3, 3)
            params[i*3 + 2] = random.uniform(0, 60)
        
        params[36] = 0.0
        params[37] = 0.0
        params[38] = 30.0
        return params

    best_result = None
    best_score = float('inf')

    # Run Simulated Annealing from multiple starting points
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        if i == 0:
            init_params = generate_initial_guess()
        elif i == 1:
            init_params = generate_random_guess()
        else:
            # Mix of structured and random
            init_params = generate_initial_guess()
            # Add some noise
            for j in range(36):
                init_params[j] += random.gauss(0, 0.2)
        
        sa_result, sa_score = simulated_annealing(init_params, max_iter=1000, T_start=1.0, T_end=0.01)
        
        if sa_score < best_score:
            best_score = sa_score
            best_result = sa_result
    
    # If SA didn't find anything good, try Nelder-Mead on the best SA result
    if best_result is not None:
        if time.time() < deadline - 20:
            try:
                res = minimize(
                    objective, 
                    best_result, 
                    method='Nelder-Mead', 
                    options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x
            except Exception as e:
                pass

    if best_result is None:
        # Fallback to a simple valid configuration
        inner = []
        inner.append((0.0, 0.0, 0.0))
        for k in range(6):
            angle = np.radians(60 * k)
            x = 2.05 * np.cos(angle)
            y = 2.05 * np.sin(angle)
            inner.append((x, y, 0.0))
        for k in range(5):
            angle = np.radians(60 * k + 30)
            x = 3.6 * np.cos(angle)
            y = 3.6 * np.sin(angle)
            inner.append((x, y, 30.0))
        oc = (0.0, 0.0)
        oa = 30.0
        os = compute_minimal_outer_side(inner, oc, oa)
        return inner, oc, os, oa

    # Extract best result
    inner = []
    for i in range(12):
        x = best_result[i*3]
        y = best_result[i*3 + 1]
        a = best_result[i*3 + 2]
        inner.append((x, y, a))
    
    oc = (best_result[36], best_result[37])
    oa = best_result[38]
    os = compute_minimal_outer_side(inner, oc, oa)
    
    return inner, oc, os, oa
