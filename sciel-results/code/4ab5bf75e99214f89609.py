import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute unit hexagon vertices for angle 0 (flat-top)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side_smooth(positions, angles, outer_angle=0.0, beta=50.0):
        """Soft-max version for smooth optimization"""
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        all_projs = []
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            all_projs.append(projs)
        
        all_projs = np.vstack(all_projs)  # (N*6, 6)
        
        # Soft-max over all vertices for each normal direction
        max_proj = 0.0
        for k in range(6):
            col = all_projs[:, k]
            # Soft max: log(sum(exp(beta*x))) / beta
            m = np.max(col)
            soft_max = m + np.log(np.sum(np.exp(beta * (col - m)))) / beta
            if soft_max > max_proj:
                max_proj = soft_max
        
        return max_proj * 2.0 / math.sqrt(3)

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def sat_gap(poly1, poly2):
        min_gap = float('inf')
        
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal1)
            if norm < 1e-9: continue
            axis = normal1 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        for i in range(6):
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal2)
            if norm < 1e-9: continue
            axis = normal2 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        return min_gap

    def is_valid(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    def compute_penalty(params, penalty_weight=100.0, use_smooth=False):
        """
        Compute the objective: outer side length + penalty for overlaps.
        params layout: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
        """
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            positions[i][0] = params[3*i]
            positions[i][1] = params[3*i+1]
            angles[i] = params[3*i+2]
            
        if use_smooth:
            side_len = calc_min_outer_side_smooth(positions, angles, 0.0, beta=30.0)
        else:
            side_len = calc_min_outer_side(positions, angles, 0.0)
        
        penalty = 0.0
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    penalty += penalty_weight * gap**2
        
        return side_len + penalty

    # Generate initial configuration using hexagonal lattice
    sqrt3 = math.sqrt(3)
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_positions = np.array(lattice_points[:12])
    init_positions -= np.mean(init_positions, axis=0)
    init_angles = np.zeros(N)
    
    # Pack into params array
    x0 = np.zeros(3*N)
    for i in range(N):
        x0[3*i] = init_positions[i][0]
        x0[3*i+1] = init_positions[i][1]
        x0[3*i+2] = init_angles[i]
    
    # Simulated annealing with adaptive step size
    best_params = x0.copy()
    best_score = compute_penalty(x0)
    current_params = x0.copy()
    current_score = best_score
    
    T = 0.5  # initial temperature
    cooling_rate = 0.9993
    step_size = 0.4
    max_iter = 80000
    
    np.random.seed(42)
    random.seed(42)
    
    for iteration in range(max_iter):
        if time.time() > deadline - 40.0:
            break
        
        # Generate perturbation
        new_params = current_params.copy()
        
        # Randomly perturb 1-2 hexagons for more focused search
        n_perturb = random.randint(1, 2)
        for _ in range(n_perturb):
            idx = random.randint(0, N-1)
            # Perturb position with larger steps early, smaller later
            pos_scale = step_size * (1.0 + 0.5 * math.exp(-iteration / 5000.0))
            new_params[3*idx] += np.random.randn() * pos_scale
            new_params[3*idx+1] += np.random.randn() * pos_scale
            # Perturb angle
            new_params[3*idx+2] += np.random.randn() * step_size * 45
        
        new_score = compute_penalty(new_params)
        
        # Acceptance criterion with occasional restarts from best
        if new_score < current_score:
            current_params = new_params
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
        else:
            delta = new_score - current_score
            if T > 0:
                acceptance_prob = math.exp(-delta / T)
                if random.random() < acceptance_prob:
                    current_params = new_params
                    current_score = new_score
        
        # Cool down
        T *= cooling_rate
        
        # Reduce step size with periodic reheating
        if iteration % 1000 == 0:
            step_size *= 0.92
        elif iteration % 5000 == 0:
            # Reheat to escape local minima
            T = max(T, 0.1)
            step_size = max(step_size, 0.15)
        
        # Occasionally jump to best and re-perturb
        if iteration % 2000 == 0 and random.random() < 0.3:
            current_params = best_params.copy()
            current_score = best_score
            # Add small random perturbation
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, N-1)
                current_params[3*idx] += np.random.randn() * 0.2
                current_params[3*idx+1] += np.random.randn() * 0.2
                current_params[3*idx+2] += np.random.randn() * 5
    
    # Check if best solution is valid
    final_positions = np.zeros((N, 2))
    final_angles = np.zeros(N)
    for i in range(N):
        final_positions[i][0] = best_params[3*i]
        final_positions[i][1] = best_params[3*i+1]
        final_angles[i] = best_params[3*i+2]
    
    if not is_valid(final_positions, final_angles, tol=1e-5):
        # Try to repair by using a simpler initial configuration
        fallback_positions = init_positions.copy()
        fallback_angles = init_angles.copy()
        final_positions = fallback_positions
        final_angles = fallback_angles
    
    # Extract best configuration
    best_positions = final_positions
    best_angles = final_angles
    
    # Multi-stage local polish with increasing precision
    if time.time() < deadline - 15.0:
        def objective_smooth(params, beta=20.0):
            positions = np.zeros((N, 2))
            angles = np.zeros(N)
            for i in range(N):
                positions[i][0] = params[3*i]
                positions[i][1] = params[3*i+1]
                angles[i] = params[3*i+2]
            
            side_len = calc_min_outer_side_smooth(positions, angles, 0.0, beta=beta)
            
            # Penalty for overlaps with soft constraint
            penalty = 0.0
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            for i in range(N):
                for j in range(i+1, N):
                    gap = sat_gap(polys[i], polys[j])
                    if gap < 0:
                        penalty += 100.0 * gap**2
            
            return side_len + penalty
        
        def objective_exact(params):
            positions = np.zeros((N, 2))
            angles = np.zeros(N)
            for i in range(N):
                positions[i][0] = params[3*i]
                positions[i][1] = params[3*i+1]
                angles[i] = params[3*i+2]
            
            side_len = calc_min_outer_side(positions, angles, 0.0)
            
            # Penalty for overlaps
            penalty = 0.0
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            for i in range(N):
                for j in range(i+1, N):
                    gap = sat_gap(polys[i], polys[j])
                    if gap < 0:
                        penalty += 200.0 * gap**2
            
            return side_len + penalty
        
        # Stage 1: Smooth optimization with moderate beta
        try:
            result1 = minimize(
                objective_smooth,
                best_params,
                method='SLSQP',
                args=(20.0,),
                options={'maxiter': 300, 'ftol': 1e-10, 'disp': False}
            )
            
            if result1.success or result1.fun < best_score:
                result_positions = np.zeros((N, 2))
                result_angles = np.zeros(N)
                for i in range(N):
                    result_positions[i][0] = result1.x[3*i]
                    result_positions[i][1] = result1.x[3*i+1]
                    result_angles[i] = result1.x[3*i+2]
                
                if is_valid(result_positions, result_angles, tol=1e-5):
                    best_positions = result_positions
                    best_angles = result_angles
                    best_params = result1.x.copy()
                    best_score = calc_min_outer_side(best_positions, best_angles, 0.0)
        except Exception:
            pass
        
        # Stage 2: Higher precision smooth optimization
        if time.time() < deadline - 10.0:
            try:
                result2 = minimize(
                    objective_smooth,
                    best_params,
                    method='SLSQP',
                    args=(50.0,),
                    options={'maxiter': 400, 'ftol': 1e-12, 'disp': False}
                )
                
                if result2.success or result2.fun < best_score:
                    result_positions = np.zeros((N, 2))
                    result_angles = np.zeros(N)
                    for i in range(N):
                        result_positions[i][0] = result2.x[3*i]
                        result_positions[i][1] = result2.x[3*i+1]
                        result_angles[i] = result2.x[3*i+2]
                    
                    if is_valid(result_positions, result_angles, tol=1e-5):
                        best_positions = result_positions
                        best_angles = result_angles
                        best_params = result2.x.copy()
                        best_score = calc_min_outer_side(best_positions, best_angles, 0.0)
            except Exception:
                pass
        
        # Stage 3: Exact objective for final polish
        if time.time() < deadline - 5.0:
            try:
                result3 = minimize(
                    objective_exact,
                    best_params,
                    method='SLSQP',
                    options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
                )
                
                if result3.success or result3.fun < best_score:
                    result_positions = np.zeros((N, 2))
                    result_angles = np.zeros(N)
                    for i in range(N):
                        result_positions[i][0] = result3.x[3*i]
                        result_positions[i][1] = result3.x[3*i+1]
                        result_angles[i] = result3.x[3*i+2]
                    
                    if is_valid(result_positions, result_angles, tol=1e-5):
                        best_positions = result_positions
                        best_angles = result_angles
                        best_params = result3.x.copy()
                        best_score = calc_min_outer_side(best_positions, best_angles, 0.0)
            except Exception:
                pass
        
        # Stage 4: Final high-precision refinement with very tight tolerance
        if time.time() < deadline - 2.0:
            try:
                result4 = minimize(
                    objective_exact,
                    best_params,
                    method='SLSQP',
                    options={'maxiter': 1000, 'ftol': 1e-15, 'disp': False}
                )
                
                if result4.success or result4.fun < best_score:
                    result_positions = np.zeros((N, 2))
                    result_angles = np.zeros(N)
                    for i in range(N):
                        result_positions[i][0] = result4.x[3*i]
                        result_positions[i][1] = result4.x[3*i+1]
                        result_angles[i] = result4.x[3*i+2]
                    
                    if is_valid(result_positions, result_angles, tol=1e-5):
                        best_positions = result_positions
                        best_angles = result_angles
                        best_params = result4.x.copy()
                        best_score = calc_min_outer_side(best_positions, best_angles, 0.0)
            except Exception:
                pass
    
    final_side = calc_min_outer_side(best_positions, best_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0
# EVOLVE-BLOCK-END
