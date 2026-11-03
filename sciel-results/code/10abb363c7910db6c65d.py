import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    # Vertices at 0, 60, 120, 180, 240, 360 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_outer_side_and_penalty(params, outer_angle=0.0, outer_center=(0.0, 0.0)):
        """
        params: [x0..x11, y0..y11, a0..a11] (36 values)
        Returns: (side_length, penalty, polys)
        side_length is the min side of outer hex.
        penalty is sum of squared overlaps.
        polys is list of vertex arrays.
        """
        pos = params[:2*N].reshape(N, 2)
        ang = params[2*N:] # radians
        
        # Outer hex is centered at outer_center and has angle outer_angle.
        # Normals for the outer hexagon.
        # Standard regular hexagon with "flat top" or "pointy top"?
        # The seed code used: normals at theta + 30 + k*60. 
        # If theta=0, normals are at 30, 90, 150, 210, 270, 330.
        # This corresponds to a hexagon with vertices at 0, 60, 120... (pointy top).
        # Wait, if normals are at 30, 90..., the sides are perpendicular to them.
        # Side at 90 normal is horizontal (y=const).
        # Side at 30 normal is slanted.
        # This is a "flat-top" hexagon? No.
        # Let's verify:
        # Vertex at 0 deg: (1, 0).
        # Edge from 0 to 60: (1,0) to (0.5, 0.866). Midpoint (0.75, 0.433).
        # Normal to this edge: (0.866, 0.5) -> angle 30 deg.
        # So yes, vertices at 0, 60... correspond to normals at 30, 90...
        
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang_n = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang_n), math.sin(ang_n)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        polys = []
        oc = np.array(outer_center)
        
        for i in range(N):
            cx, cy = pos[i]
            a_rad = ang[i]
            verts = get_hex_verts(cx, cy, 1.0, np.degrees(a_rad))
            polys.append(verts)
            verts_rel = verts - oc
            projs = verts_rel @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        side = max_proj * 2.0 / SQRT3
        
        # Calculate overlap penalty
        # Use SAT to check overlap depth.
        penalty = 0.0
        for i in range(N):
            p1 = polys[i]
            for j in range(i+1, N):
                p2 = polys[j]
                
                # Quick bounding box check
                min_x1, min_y1 = np.min(p1, axis=0)
                max_x1, max_y1 = np.max(p1, axis=0)
                min_x2, min_y2 = np.min(p2, axis=0)
                max_x2, max_y2 = np.max(p2, axis=0)
                
                if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                    continue # No overlap possible
                
                # SAT
                min_sep = float('inf')
                axes = []
                for k in range(6):
                    edge = p1[(k+1)%6] - p1[k]
                    n = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(n)
                    if norm > 1e-9: axes.append(n/norm)
                for k in range(6):
                    edge = p2[(k+1)%6] - p2[k]
                    n = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(n)
                    if norm > 1e-9: axes.append(n/norm)
                
                for axis in axes:
                    proj1 = np.dot(p1, axis)
                    proj2 = np.dot(p2, axis)
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    # Separation: positive if apart, negative if overlapping
                    sep = max(min1, min2) - min(max1, max2)
                    if sep < min_sep:
                        min_sep = sep
                
                if min_sep < 0:
                    # Overlap depth is -min_sep
                    penalty += (-min_sep) ** 2 * 10.0
        
        return side, penalty, polys

    def objective(params):
        side, penalty, _ = calc_outer_side_and_penalty(params)
        return side + penalty

    def polish(params):
        """
        High-precision refinement using SLSQP on the current best configuration.
        """
        def obj_p(params):
            side, penalty, _ = calc_outer_side_and_penalty(params)
            return side + penalty
        
        # Constraints: No overlap.
        # We use soft constraints via penalty in objective for SLSQP? 
        # SLSQP requires explicit constraints.
        # Defining 66 constraints (12*11/2) is heavy but N=12 is small.
        # However, defining the constraint function is expensive.
        # Let's just use a very tight Nelder-Mead or COBYLA for polishing, 
        # or a simple gradient-free step.
        # Actually, let's use a simple local search with smaller steps.
        
        best_p = params.copy()
        best_f = obj_p(best_p)
        
        # Simple local search
        step = 0.01
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            for k in range(len(best_p)):
                # Try small perturbations
                for delta in [step, -step]:
                    trial = best_p.copy()
                    trial[k] += delta
                    f_trial = obj_p(trial)
                    if f_trial < best_f:
                        best_p = trial
                        best_f = f_trial
                        improved = True
            if not improved:
                step *= 0.7
                if step < 1e-5:
                    break
                    
        return best_p

    # Initial configurations
    inits = []
    
    # 1. Honeycomb lattice (standard)
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * 1.5
            y = i * SQRT3 / 2.0 + j * SQRT3
            lattice_points.append((x, y))
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_pos = np.array(lattice_points[:12])
    init_pos -= np.mean(init_pos, axis=0)
    init_ang = np.zeros(N)
    inits.append(np.concatenate([init_pos.flatten(), init_ang]))
    
    # 2. Ring configuration
    ring_pos = []
    ring_ang = []
    r = 2.0
    for i in range(N):
        a = 2 * math.pi * i / N
        ring_pos.append([r * math.cos(a), r * math.sin(a)])
        ring_ang.append(a)
    ring_pos = np.array(ring_pos)
    ring_pos -= np.mean(ring_pos, axis=0)
    inits.append(np.concatenate([ring_pos.flatten(), np.array(ring_ang)]))
    
    # 3. Random starts
    np.random.seed(42)
    for _ in range(5):
        pos = np.random.uniform(-3, 3, (N, 2))
        ang = np.random.uniform(0, 2*math.pi, N)
        inits.append(np.concatenate([pos.flatten(), ang]))

    best_side = float('inf')
    best_params = None
    best_outer_center = [0.0, 0.0]
    best_outer_angle = 0.0

    # Simulated Annealing / Local Search Strategy
    # We will run a simple SA on each initial configuration.
    
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        current = x0.copy()
        current_side, current_penalty, _ = calc_outer_side_and_penalty(current)
        current_obj = current_side + current_penalty
        
        best_local = current.copy()
        best_local_obj = current_obj
        
        # Annealing schedule
        T = 1.0
        cooling = 0.95
        max_iters = 5000
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Perturbation
            trial = current.copy()
            # Move a random hexagon
            i = np.random.randint(N)
            dx = np.random.normal(0, 0.1)
            dy = np.random.normal(0, 0.1)
            da = np.random.normal(0, 0.1)
            
            trial[2*i] += dx
            trial[2*i+1] += dy
            trial[2*N+i] += da
            
            trial_side, trial_penalty, _ = calc_outer_side_and_penalty(trial)
            trial_obj = trial_side + trial_penalty
            
            delta = trial_obj - current_obj
            
            if delta < 0 or np.random.rand() < math.exp(-delta / T):
                current = trial
                current_obj = trial_obj
                
                if current_obj < best_local_obj:
                    best_local = current.copy()
                    best_local_obj = current_obj
            
            T *= cooling
            
            # If T is too small, break
            if T < 0.001:
                break
        
        # Polish the best local found
        if best_local_obj < best_side:
            polished = polish(best_local)
            pol_side, pol_penalty, _ = calc_outer_side_and_penalty(polished)
            pol_obj = pol_side + pol_penalty
            
            if pol_obj < best_side and pol_penalty < 1e-4:
                best_side = pol_side
                best_params = polished.copy()
        
        # Also check if the raw best_local is better (in case polish failed)
        if best_local_obj < best_side:
            bl_side, bl_penalty, _ = calc_outer_side_and_penalty(best_local)
            if bl_penalty < 1e-4 and bl_side < best_side:
                best_side = bl_side
                best_params = best_local.copy()

    # Final Polish: Optimize outer center and angle for the best inner config
    if best_params is not None:
        pos = best_params[:2*N].reshape(N, 2)
        ang = best_params[2*N:]
        
        def outer_obj(outer_params):
            oc = outer_params[:2]
            oa = outer_params[2] # radians
            theta_rad = oa
            normals = []
            for k in range(6):
                ang_n = theta_rad + math.pi/6 + k * math.pi/3
                normals.append([math.cos(ang_n), math.sin(ang_n)])
            normals = np.array(normals)
            
            max_proj = 0.0
            for i in range(N):
                cx, cy = pos[i]
                a_rad = ang[i]
                verts = get_hex_verts(cx, cy, 1.0, np.degrees(a_rad))
                verts_rel = verts - np.array(oc)
                projs = verts_rel @ normals.T
                max_proj_i = np.max(projs)
                if max_proj_i > max_proj:
                    max_proj = max_proj_i
            
            return max_proj * 2.0 / SQRT3

        centroid = np.mean(pos, axis=0)
        x0_outer = np.array([centroid[0], centroid[1], 0.0])
        
        try:
            res_outer = minimize(
                outer_obj, 
                x0_outer, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res_outer.fun < best_side:
                best_side = res_outer.fun
                best_outer_center = res_outer.x[:2]
                best_outer_angle = np.degrees(res_outer.x[2])
        except:
            pass

    # Fallback
    if best_params is None:
        sqrt3 = math.sqrt(3)
        coords = [
            (-2, 0), (0, 0), (2, 0),
            (-1, sqrt3), (1, sqrt3), (3, sqrt3),
            (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
            (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
        ]
        cx = sum(c[0] for c in coords) / N
        cy = sum(c[1] for c in coords) / N
        pos = np.array([[c[0]-cx, c[1]-cy] for c in coords])
        ang = np.zeros(N)
        best_params = np.concatenate([pos.flatten(), ang])
        best_side, _, _ = calc_outer_side_and_penalty(best_params)
        best_outer_center = [0.0, 0.0]
        best_outer_angle = 0.0

    # Construct return values
    final_pos = best_params[:2*N].reshape(N, 2)
    final_ang = best_params[2*N:]
    
    inner = []
    for i in range(N):
        inner.append([final_pos[i][0], final_pos[i][1], np.degrees(final_ang[i])])
    
    return inner, tuple(best_outer_center), best_side, best_outer_angle
