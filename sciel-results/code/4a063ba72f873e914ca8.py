import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def _hexagon_vertices(cx, cy, angle, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    angles = angle + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _hexagon_overlap(v1, v2, tol=1e-9):
    """Check if two hexagons (as vertex arrays) overlap using SAT."""
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    
    all_edges = np.vstack([edges1, edges2])
    
    for edge in all_edges:
        nx, ny = -edge[1], edge[0]
        norm = math.hypot(nx, ny)
        if norm < 1e-12:
            continue
        nx, ny = nx / norm, ny / norm
        
        proj1 = v1[:, 0] * nx + v1[:, 1] * ny
        proj2 = v2[:, 0] * nx + v2[:, 1] * ny
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 + tol < min2 or max2 + tol < min1:
            return False
    
    return True

def _force_directed_layout(n_hex, max_iter=500, dt=0.01, repulse_strength=5.0, attract_strength=0.5, seed=None):
    """
    Force-directed layout to find a compact arrangement of n_hex unit hexagons.
    Returns array of (x, y) positions.
    """
    if seed is not None:
        np.random.seed(seed)
        
    positions = np.zeros((n_hex, 2))
    idx = 0
    for r in range(-3, 4):
        for q in range(-3, 4):
            if idx >= n_hex:
                break
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            positions[idx] = [x, y]
            idx += 1
        if idx >= n_hex:
            break
    
    positions += np.random.uniform(-0.2, 0.2, size=(n_hex, 2))
    
    for iteration in range(max_iter):
        forces = np.zeros((n_hex, 2))
        
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-9:
                    diff = np.random.uniform(-0.01, 0.01, size=2)
                    dist = np.linalg.norm(diff)
                
                if dist < 2.5:
                    force_mag = repulse_strength / (dist ** 2)
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist
        
        centroid = np.mean(positions, axis=0)
        for i in range(n_hex):
            forces[i] += -attract_strength * (positions[i] - centroid)
        
        positions += dt * forces
        dt *= 0.995
        
    return positions

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate diverse initial candidates
    candidates = []
    
    # 1. Axial lattice candidates (kept for robustness)
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)
        
    candidates.append(axial_to_cartesian(axial_1))
    candidates.append(axial_to_cartesian(axial_2))
    candidates.append(axial_to_cartesian(axial_3))
    
    # 2. Force-directed candidates with varied parameters
    for trial in range(20):
        seed = 100 + trial
        repulse = 3.0 + (trial % 5) * 0.5
        attract = 0.2 + (trial % 3) * 0.1
        max_iter = 300 + (trial % 4) * 100
        pos = _force_directed_layout(N, max_iter=max_iter, dt=0.02, 
                                      repulse_strength=repulse, 
                                      attract_strength=attract, seed=seed)
        # Center at origin
        pos -= np.mean(pos, axis=0)
        candidates.append(pos)

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        n_pts = len(pts)
        A_ub_list = []
        b_ub_list = []
        
        for nx, ny in normals:
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            A_ub_list.append(np.column_stack([-nx * np.ones(n_pts), -ny * np.ones(n_pts), -np.ones(n_pts)]))
            b_ub_list.append(-(projs))
            A_ub_list.append(np.column_stack([nx * np.ones(n_pts), ny * np.ones(n_pts), -np.ones(n_pts)]))
            b_ub_list.append(projs)
        
        A_ub = np.vstack(A_ub_list)
        b_ub = np.concatenate(b_ub_list)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(np.array([0.0, 0.0, 1.0]), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        
        max_val = np.max(all_projs)
        if max_val > 100:
            return float('inf'), (cx, cy), float('inf')
        
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy), D

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search
        for i in range(73):
            beta = i * (math.pi / 6) / 72
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 500, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # Smooth surrogate optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
                start_betas = [best_beta, best_beta + 0.08, best_beta - 0.08, 0.0, math.pi/12, math.pi/6, best_beta + 0.04, best_beta - 0.04]
                
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        
                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S, _, _ = smooth_calc_S(pts, beta, cx, cy, temperature=t)
                            return S
                        
                        init = [best_center[0], best_center[1], sb]
                        res = minimize(smooth_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 800, 'xatol': 1e-11, 'fatol': 1e-11})
                        
                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = c_exact
        except Exception:
            pass
            
        # Final high-precision polish
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
                    normals = np.column_stack([np.cos(angles), np.sin(angles)])
                    
                    centered = pts - np.array([cx, cy])
                    projs = np.abs(centered @ normals.T)
                    max_proj = np.max(projs)
                    
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S
                
                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.03],
                    [best_center[0], best_center[1], best_beta - 0.03],
                    [best_center[0] + 0.03, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.03, best_beta],
                    [best_center[0] - 0.03, best_center[1], best_beta],
                    [best_center[0], best_center[1] - 0.03, best_beta],
                ]
                
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-14})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Process all candidates
    # Sort by bounding box area
    def bbox_heuristic(pts):
        extent_x = np.max(pts[:, 0]) - np.min(pts[:, 0])
        extent_y = np.max(pts[:, 1]) - np.min(pts[:, 1])
        return extent_x * extent_y
    
    sorted_candidates = sorted(candidates, key=bbox_heuristic)
    
    for pts in sorted_candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = sorted_candidates[0]
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    # Second pass: Continuous optimization of inner positions with smooth overlap penalty
    if best_global_S < float('inf') and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        
        def smooth_overlap_penalty(pts, beta, temperature=0.05):
            """Compute smooth overlap penalty using log-sum-exp of penetration depths."""
            n_pts = len(pts)
            penalty_terms = []
            
            for i in range(n_pts):
                for j in range(i+1, n_pts):
                    dx = pts[i,0] - pts[j,0]
                    dy = pts[i,1] - pts[j,1]
                    dist_sq = dx*dx + dy*dy
                    if dist_sq > 4.0:
                        continue
                    
                    # Compute minimum penetration depth using SAT
                    v1 = _hexagon_vertices(pts[i,0], pts[i,1], beta)
                    v2 = _hexagon_vertices(pts[j,0], pts[j,1], beta)
                    
                    edges1 = np.roll(v1, -1, axis=0) - v1
                    edges2 = np.roll(v2, -1, axis=0) - v2
                    all_edges = np.vstack([edges1, edges2])
                    
                    min_penetration = float('inf')
                    for edge in all_edges:
                        nx, ny = -edge[1], edge[0]
                        norm = math.hypot(nx, ny)
                        if norm < 1e-12:
                            continue
                        nx, ny = nx / norm, ny / norm
                        
                        proj1 = v1[:, 0] * nx + v1[:, 1] * ny
                        proj2 = v2[:, 0] * nx + v2[:, 1] * ny
                        
                        min1, max1 = np.min(proj1), np.max(proj1)
                        min2, max2 = np.min(proj2), np.max(proj2)
                        
                        # Penetration depth is how much the intervals overlap
                        # If max1 < min2 or max2 < min1, no overlap on this axis
                        if max1 < min2 or max2 < min1:
                            min_penetration = 0.0
                            break
                        else:
                            # Overlap depth = min(max1, max2) - max(min1, min2)
                            overlap = min(max1, max2) - max(min1, min2)
                            min_penetration = min(min_penetration, overlap)
                    
                    if min_penetration > 0:
                        # Convert to penalty: large when penetrating
                        penalty_terms.append(min_penetration)
            
            if not penalty_terms:
                return 0.0
            
            # Smooth sum using log-sum-exp
            arr = np.array(penalty_terms)
            max_val = np.max(arr)
            if max_val > 10:
                return 1000.0
            shifted = (arr - max_val) / temperature
            shifted = np.clip(shifted, -50, 50)
            log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
            return temperature * log_sum * 100.0

        def continuous_objective(params):
            # params: [x1, y1, ..., xN, yN, cx, cy, beta]
            n_pts = N
            pts = params[:2*n_pts].reshape(n_pts, 2)
            cx = params[2*n_pts]
            cy = params[2*n_pts+1]
            beta = params[2*n_pts+2]
            
            # Normalize beta to [0, pi/6)
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            
            # Compute S
            angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            
            centered = pts - np.array([cx, cy])
            projs = np.abs(centered @ normals.T)
            max_proj = np.max(projs)
            
            S = (max_proj + APOTHEM) / APOTHEM
            
            # Smooth overlap penalty
            penalty = smooth_overlap_penalty(pts, beta, temperature=0.05)
            
            return S + penalty

        # Initial guess: best_global_pts, best_global_center, best_global_beta
        init_params = list(best_global_pts.flatten()) + [best_global_center[0], best_global_center[1], best_global_beta]
        
        # Optimize with multiple strategies
        best_candidate = None
        best_candidate_S = float('inf')
        
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                # Strategy 1: Nelder-Mead with smooth penalty
                res = minimize(continuous_objective, init_params, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-11})
                
                new_pts = res.x[:2*N].reshape(N, 2)
                new_cx = res.x[2*N]
                new_cy = res.x[2*N+1]
                new_beta = ((res.x[2*N+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                
                # Verify no overlap
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        v1 = _hexagon_vertices(new_pts[i,0], new_pts[i,1], new_beta)
                        v2 = _hexagon_vertices(new_pts[j,0], new_pts[j,1], new_beta)
                        if _hexagon_overlap(v1, v2, tol=1e-9):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    angles = new_beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
                    normals = np.column_stack([np.cos(angles), np.sin(angles)])
                    centered = new_pts - np.array([new_cx, new_cy])
                    projs = np.abs(centered @ normals.T)
                    max_proj = np.max(projs)
                    new_S = (max_proj + APOTHEM) / APOTHEM
                    
                    if new_S < best_candidate_S:
                        best_candidate_S = new_S
                        best_candidate = (new_pts, new_cx, new_cy, new_beta)
                        
        except Exception:
            pass
        
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_candidate is not None:
                # Strategy 2: Powell method for further refinement
                def objective_powell(params):
                    n_pts = N
                    pts = params[:2*n_pts].reshape(n_pts, 2)
                    cx = params[2*n_pts]
                    cy = params[2*n_pts+1]
                    beta = params[2*n_pts+2]
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    
                    angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
                    normals = np.column_stack([np.cos(angles), np.sin(angles)])
                    centered = pts - np.array([cx, cy])
                    projs = np.abs(centered @ normals.T)
                    max_proj = np.max(projs)
                    S = (max_proj + APOTHEM) / APOTHEM
                    penalty = smooth_overlap_penalty(pts, beta, temperature=0.03)
                    return S + penalty
                
                init2 = list(best_candidate[0].flatten()) + [best_candidate[1], best_candidate[2], best_candidate[3]]
                res2 = minimize(objective_powell, init2, method='Powell',
                              options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
                
                new_pts = res2.x[:2*N].reshape(N, 2)
                new_cx = res2.x[2*N]
                new_cy = res2.x[2*N+1]
                new_beta = ((res2.x[2*N+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        v1 = _hexagon_vertices(new_pts[i,0], new_pts[i,1], new_beta)
                        v2 = _hexagon_vertices(new_pts[j,0], new_pts[j,1], new_beta)
                        if _hexagon_overlap(v1, v2, tol=1e-9):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    angles = new_beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
                    normals = np.column_stack([np.cos(angles), np.sin(angles)])
                    centered = new_pts - np.array([new_cx, new_cy])
                    projs = np.abs(centered @ normals.T)
                    max_proj = np.max(projs)
                    new_S = (max_proj + APOTHEM) / APOTHEM
                    
                    if new_S < best_candidate_S:
                        best_candidate_S = new_S
                        best_candidate = (new_pts, new_cx, new_cy, new_beta)
                        
        except Exception:
            pass
        
        if best_candidate is not None and best_candidate_S < best_global_S:
            best_global_S = best_candidate_S
            best_global_pts = best_candidate[0]
            best_global_center = (best_candidate[1], best_candidate[2])
            best_global_beta = best_candidate[3]

    # Third pass: High-precision polish with epigraph-style formulation
    # Use a smooth max approximation and tight tolerances for final refinement
    
    if best_global_S < float('inf') and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        
        def polish_objective(params):
            n_pts = N
            pts = params[:2*n_pts].reshape(n_pts, 2)
            cx = params[2*n_pts]
            cy = params[2*n_pts+1]
            beta = params[2*n_pts+2]
            
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            
            angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            
            centered = pts - np.array([cx, cy])
            projs = np.abs(centered @ normals.T)
            
            # Smooth max via log-sum-exp
            temperature = 0.01
            max_val = np.max(projs)
            if max_val > 100:
                return float('inf')
            shifted = (projs.ravel() - max_val) / temperature
            shifted = np.clip(shifted, -50, 50)
            log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
            D = temperature * log_sum
            
            S = (D + APOTHEM) / APOTHEM
            
            # Light smooth overlap penalty
            penalty = smooth_overlap_penalty(pts, beta, temperature=0.02)
            
            return S + penalty

        init_params = list(best_global_pts.flatten()) + [best_global_center[0], best_global_center[1], best_global_beta]
        
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                # Multiple starting points for robustness
                start_offsets = [
                    [0, 0, 0],
                    [0.01, 0, 0], [-0.01, 0, 0],
                    [0, 0.01, 0], [0, -0.01, 0],
                    [0, 0, 0.01], [0, 0, -0.01],
                ]
                
                for offset in start_offsets:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    
                    init = list(best_global_pts.flatten()) + [
                        best_global_center[0] + offset[0],
                        best_global_center[1] + offset[1],
                        best_global_beta + offset[2]
                    ]
                    
                    res = minimize(polish_objective, init, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
                    
                    new_pts = res.x[:2*N].reshape(N, 2)
                    new_cx = res.x[2*N]
                    new_cy = res.x[2*N+1]
                    new_beta = ((res.x[2*N+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    
                    # Verify validity
                    valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            v1 = _hexagon_vertices(new_pts[i,0], new_pts[i,1], new_beta)
                            v2 = _hexagon_vertices(new_pts[j,0], new_pts[j,1], new_beta)
                            if _hexagon_overlap(v1, v2, tol=1e-9):
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Recompute exact S
                        angles = new_beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
                        normals = np.column_stack([np.cos(angles), np.sin(angles)])
                        centered = new_pts - np.array([new_cx, new_cy])
                        projs = np.abs(centered @ normals.T)
                        max_proj = np.max(projs)
                        new_S = (max_proj + APOTHEM) / APOTHEM
                        
                        if new_S < best_global_S:
                            best_global_S = new_S
                            best_global_pts = new_pts
                            best_global_center = (new_cx, new_cy)
                            best_global_beta = new_beta
        except Exception:
            pass

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Safety scaling
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
