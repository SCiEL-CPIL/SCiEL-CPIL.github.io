"""
Strategy: 6-fold symmetric honeycomb packing with dual optimization formulations.
Optimized: fast disjointness check, no disjointness in smooth objective,
consolidated polish phases with coordinate descent + Nelder-Mead.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin before 150s kill

    # Precompute unit hexagon vertices at angle 0
    unit_hex_verts = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))]
        for a in range(0, 360, 60)
    ])
    # Precompute edge normals for unit hexagon
    unit_edge_normals = np.array([
        [-math.sin(math.radians(60 * k + 30)), math.cos(math.radians(60 * k + 30))]
        for k in range(6)
    ])

    def get_inner_positions(r1, a1, r2, a2):
        """Generate 12 inner hexagon positions: (x, y, angle_deg)."""
        positions = []
        positions.append((0.0, 0.0, 0.0))
        for k in range(6):
            angle = math.radians(60 * k)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            positions.append((x, y, a1))
        for k in range(5):
            angle = math.radians(60 * k + 30)
            x = r2 * math.cos(angle)
            y = r2 * math.sin(angle)
            positions.append((x, y, a2))
        return positions

    def compute_clearances(r1, a1, r2, a2, outer_angle_deg):
        """Compute the 6 clearance values (one per outer normal)."""
        outer_angle_rad = math.radians(outer_angle_deg)
        normal_angles = outer_angle_rad + math.radians(30) + math.radians(60) * np.arange(6)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        positions = get_inner_positions(r1, a1, r2, a2)
        
        clearances = np.zeros(6)
        for (cx, cy, alpha_deg) in positions:
            alpha_rad = math.radians(alpha_deg)
            cos_a = math.cos(alpha_rad)
            sin_a = math.sin(alpha_rad)
            local_normals = np.column_stack([
                normals[:, 0] * cos_a + normals[:, 1] * sin_a,
                -normals[:, 0] * sin_a + normals[:, 1] * cos_a
            ])
            dots = local_normals @ unit_hex_verts.T  # (6, 6)
            supports = np.max(dots, axis=1)  # (6,)
            center_proj = cx * normals[:, 0] + cy * normals[:, 1]  # (6,)
            vals = center_proj + supports  # (6,)
            clearances = np.maximum(clearances, vals)
        
        return clearances

    def compute_min_side_fast(r1, a1, r2, a2, outer_angle_deg):
        """Compute minimal outer hexagon side length."""
        clearances = compute_clearances(r1, a1, r2, a2, outer_angle_deg)
        apothem = np.max(clearances)
        side = apothem / (math.sqrt(3) / 2)
        return side

    def check_disjoint_fast(r1, a1, r2, a2, tol=1e-9):
        """Check if inner hexagons are pairwise disjoint using SAT (optimized)."""
        positions = get_inner_positions(r1, a1, r2, a2)
        n_pos = len(positions)
        
        # Precompute all vertices and edge normals
        all_verts = np.zeros((n_pos, 6, 2))
        all_normals = np.zeros((n_pos, 6, 2))
        
        for i, (xi, yi, ai) in enumerate(positions):
            ai_rad = math.radians(ai)
            cos_a = math.cos(ai_rad)
            sin_a = math.sin(ai_rad)
            # Rotate unit hex vertices
            for k in range(6):
                vx = unit_hex_verts[k, 0]
                vy = unit_hex_verts[k, 1]
                all_verts[i, k, 0] = xi + vx * cos_a - vy * sin_a
                all_verts[i, k, 1] = yi + vx * sin_a + vy * cos_a
            # Rotate unit edge normals
            for k in range(6):
                nx = unit_edge_normals[k, 0]
                ny = unit_edge_normals[k, 1]
                all_normals[i, k, 0] = nx * cos_a - ny * sin_a
                all_normals[i, k, 1] = nx * sin_a + ny * cos_a
        
        for i in range(n_pos):
            for j in range(i + 1, n_pos):
                vi = all_verts[i]  # (6, 2)
                vj = all_verts[j]  # (6, 2)
                # Combined axes: 12 normals
                axes = np.vstack([all_normals[i], all_normals[j]])  # (12, 2)
                
                # Project all vertices onto all axes: (12, 6) for each
                proj_i = vi @ axes.T  # (6, 12)
                proj_j = vj @ axes.T  # (6, 12)
                
                # Check separation on any axis
                max_i = proj_i.max(axis=0)  # (12,)
                min_i = proj_i.min(axis=0)  # (12,)
                max_j = proj_j.max(axis=0)  # (12,)
                min_j = proj_j.min(axis=0)  # (12,)
                
                # Separated if max_i < min_j - tol OR max_j < min_i - tol on any axis
                if np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol):
                    continue
                else:
                    return False
        return True

    def objective_epigraph(params, beta=80.0):
        """Epigraph formulation: minimize t subject to smooth clearance <= t.
        params = [r1, a1, r2, a2, o_a, t]
        """
        if time.time() > deadline - 5:
            return 1e6
        r1, a1, r2, a2, o_a, t = params
        if r1 < 0.5 or r2 < 0.5 or t < 0.5:
            return 1e6
        
        clearances = compute_clearances(r1, a1, r2, a2, o_a)
        violations = np.maximum(0, clearances - t)
        smooth_violation = np.sum(np.log1p(np.exp(beta * violations))) / beta
        side = t / (math.sqrt(3) / 2)
        return side + 50.0 * smooth_violation

    def objective_smooth(params, beta=50.0):
        """Smooth objective using soft-max of clearances (no disjointness check)."""
        if time.time() > deadline - 5:
            return 1e6
        r1, a1, r2, a2, o_a = params
        if r1 < 0.5 or r2 < 0.5:
            return 1e6
        
        clearances = compute_clearances(r1, a1, r2, a2, o_a)
        scaled = beta * clearances
        max_val = np.max(scaled)
        soft_max_val = max_val / beta + math.log(np.sum(np.exp(scaled - max_val))) / beta
        side = soft_max_val / (math.sqrt(3) / 2)
        return side

    def objective_exact(params):
        """Exact objective with disjointness penalty."""
        if time.time() > deadline - 5:
            return 1e6
        r1, a1, r2, a2, o_a = params
        if r1 < 0.5 or r2 < 0.5:
            return 1e6
        if not check_disjoint_fast(r1, a1, r2, a2):
            return 50.0 + compute_min_side_fast(r1, a1, r2, a2, o_a)
        return compute_min_side_fast(r1, a1, r2, a2, o_a)

    # Multi-start global search
    best_side = float('inf')
    best_params = None
    candidate_list = []  # Store all candidates for later polishing
    
    # Initial guesses based on honeycomb geometry
    initial_guesses = [
        [2.0, 0.0, 2*math.sqrt(3), 0.0, 30.0],
        [2.0, 0.0, 2*math.sqrt(3), 30.0, 0.0],
        [2.0, 30.0, 2*math.sqrt(3), 0.0, 0.0],
        [2.0, 0.0, 2*math.sqrt(3), 0.0, 0.0],
        [2.0, 0.0, 2*math.sqrt(3), 30.0, 30.0],
        [1.95, 0.0, 3.45, 0.0, 30.0],
        [2.05, 0.0, 3.5, 0.0, 30.0],
        [1.98, 10.0, 3.48, 20.0, 15.0],
        [2.02, 5.0, 3.52, 10.0, 25.0],
        [1.92, 15.0, 3.42, 45.0, 10.0],
        [2.08, 25.0, 3.55, 15.0, 40.0],
    ]
    
    np.random.seed(42)
    for _ in range(12):
        r1 = np.random.uniform(1.85, 2.15)
        a1 = np.random.uniform(0, 60)
        r2 = np.random.uniform(3.3, 3.65)
        a2 = np.random.uniform(0, 60)
        o_a = np.random.uniform(0, 60)
        initial_guesses.append([r1, a1, r2, a2, o_a])
    
    bounds_5d = [(1.5, 2.5), (0, 60), (2.5, 4.5), (0, 60), (0, 60)]
    bounds_6d = bounds_5d + [(0.5, 5.0)]
    
    # Phase 1: L-BFGS-B on epigraph formulation for all initial guesses
    for idx, guess in enumerate(initial_guesses):
        if time.time() > deadline - 40:
            break
        
        try:
            t_init = compute_min_side_fast(*guess) * (math.sqrt(3) / 2) * 1.05
            guess_6d = guess + [t_init]
            result = minimize(
                objective_epigraph,
                x0=np.array(guess_6d, dtype=float),
                method='L-BFGS-B',
                bounds=bounds_6d,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 1000}
            )
            if result.success or result.fun < 50:
                params_5d = result.x[:5]
                exact_val = compute_min_side_fast(*params_5d)
                if check_disjoint_fast(*params_5d[:4]) and exact_val < best_side:
                    best_side = exact_val
                    best_params = params_5d.copy()
                # Store candidate for later polishing
                if result.fun < 50:
                    candidate_list.append(params_5d.copy())
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish on top candidates with smooth objective
    if time.time() < deadline - 30:
        # Sort candidates by their exact side length
        scored_candidates = []
        for cand in candidate_list:
            try:
                val = compute_min_side_fast(*cand)
                if check_disjoint_fast(*cand[:4]):
                    scored_candidates.append((val, cand))
            except:
                pass
        scored_candidates.sort(key=lambda x: x[0])
        
        # Polish top 5 candidates
        for val, cand in scored_candidates[:5]:
            if time.time() > deadline - 25:
                break
            try:
                result2 = minimize(
                    objective_smooth,
                    x0=np.array(cand, dtype=float),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                if result2.fun < 50:
                    params = result2.x
                    exact_val = compute_min_side_fast(*params)
                    if check_disjoint_fast(*params[:4]) and exact_val < best_side:
                        best_side = exact_val
                        best_params = params.copy()
            except Exception:
                pass
    
    # Phase 3: Coordinate descent polish on best_params
    if best_params is not None and time.time() < deadline - 15:
        try:
            x = best_params.copy()
            step_sizes = [0.01, 0.005, 0.001, 0.0005, 0.0001]
            for step in step_sizes:
                if time.time() > deadline - 12:
                    break
                improved = True
                while improved and time.time() < deadline - 12:
                    improved = False
                    for dim in range(5):
                        for direction in [1.0, -1.0]:
                            x_try = x.copy()
                            x_try[dim] += direction * step
                            if x_try[0] < 1.5 or x_try[0] > 2.5:
                                continue
                            if x_try[2] < 2.5 or x_try[2] > 4.5:
                                continue
                            try:
                                val_try = compute_min_side_fast(*x_try)
                                if check_disjoint_fast(*x_try[:4]) and val_try < best_side:
                                    best_side = val_try
                                    x = x_try.copy()
                                    improved = True
                            except:
                                pass
            best_params = x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision Nelder-Mead polish
    if best_params is not None and time.time() < deadline - 8:
        try:
            result_final = minimize(
                objective_exact,
                x0=best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16}
            )
            if result_final.fun < 50:
                params = result_final.x
                exact_val = compute_min_side_fast(*params)
                if check_disjoint_fast(*params[:4]) and exact_val < best_side:
                    best_side = exact_val
                    best_params = params.copy()
        except Exception:
            pass
    
    # Phase 5: Ultra-precise final polish
    if best_params is not None and time.time() < deadline - 4:
        try:
            result_ultra = minimize(
                objective_exact,
                x0=best_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-15, 'fatol': 1e-17}
            )
            if result_ultra.fun < 50:
                params = result_ultra.x
                exact_val = compute_min_side_fast(*params)
                if check_disjoint_fast(*params[:4]) and exact_val < best_side:
                    best_side = exact_val
                    best_params = params.copy()
        except Exception:
            pass
    
    # If no valid solution found, use a known valid configuration
    if best_params is None:
        best_params = np.array([2.0, 0.0, 2*math.sqrt(3), 0.0, 30.0])
        best_side = compute_min_side_fast(*best_params)
    
    r1, a1, r2, a2, o_a = best_params
    a1 = a1 % 360
    a2 = a2 % 360
    o_a = o_a % 360
    
    inner = []
    inner.append((0.0, 0.0, 0.0))
    for k in range(6):
        angle = math.radians(60 * k)
        x = r1 * math.cos(angle)
        y = r1 * math.sin(angle)
        inner.append((x, y, a1))
    for k in range(5):
        angle = math.radians(60 * k + 30)
        x = r2 * math.cos(angle)
        y = r2 * math.sin(angle)
        inner.append((x, y, a2))
    
    oc = (0.0, 0.0)
    oa = o_a
    os = compute_min_side_fast(r1, a1, r2, a2, o_a)
    
    return inner, oc, os, oa
