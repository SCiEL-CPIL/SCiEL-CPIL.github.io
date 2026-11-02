import math
import numpy as np
import time
from scipy.optimize import linprog, minimize
from itertools import product

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # 1. Generate candidate topologies (hexagonal polyominoes of size N)
    # We use a simple DFS to generate connected sets of N cells on a hex grid.
    # We prune by bounding box size to keep only compact shapes.
    
    def generate_topologies(n_cells):
        topologies = set()
        
        def canonicalize(cells):
            # Translate to min x, min y
            min_q = min(q for q, r in cells)
            min_r = min(r for q, r in cells)
            shifted = frozenset((q - min_q, r - min_r) for q, r in cells)
            return shifted

        def dfs(cells, remaining, last):
            if len(cells) == n_cells:
                topologies.add(canonicalize(cells))
                return
            
            q, r = last
            # Hex neighbors in axial coordinates
            neighbors = [
                (q+1, r), (q-1, r),
                (q, r+1), (q, r-1),
                (q+1, r-1), (q-1, r+1)
            ]
            
            for nq, nr in neighbors:
                if (nq, nr) not in cells:
                    # Pruning: Check bounding box to avoid large sparse shapes
                    # A compact shape of 11 hexagons should have a small span
                    qs = [c[0] for c in cells] + [nq]
                    rs = [c[1] for c in cells] + [nr]
                    span_q = max(qs) - min(qs)
                    span_r = max(rs) - min(rs)
                    # Heuristic: span should not be too large relative to area
                    # For 11 cells, a "line" would have span ~10. A "blob" should have span < 5
                    if span_q > 6 or span_r > 6:
                        continue
                    
                    dfs(cells | {(nq, nr)}, remaining - 1, (nq, nr))
        
        dfs({(0, 0)}, n_cells - 1, (0, 0))
        return list(topologies)

    # Generate topologies
    raw_topologies = generate_topologies(N)
    
    # Filter and sort by compactness (bounding box area)
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def bounding_box_area(pts):
        if len(pts) == 0: return float('inf')
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        return (max_x - min_x) * (max_y - min_y)

    scored_topologies = []
    for topo in raw_topologies:
        pts = axial_to_cartesian(list(topo))
        area = bounding_box_area(pts)
        scored_topologies.append((area, topo))
        
    # Sort by area (most compact first)
    scored_topologies.sort(key=lambda x: x[0])
    
    # Take the top K most compact topologies to balance search time
    K_CANDIDATES = 100
    candidates = [topo for _, topo in scored_topologies[:K_CANDIDATES]]
    
    # Fallback: if generation failed or is empty, use a known good one
    if not candidates:
        candidates = [frozenset([(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,1), (-1,2), (-2,1), (2,-1)])]

    # 2. Optimization Functions
    
    def calc_S_for_beta_fast(pts, beta):
        """Fast LP-based calculation of minimum enclosing hexagon side length."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        # Variables: cx, cy, D
        # Minimize D
        c_obj = np.array([0.0, 0.0, 1.0])
        
        # Constraints:
        # For each normal n_i and point p_j:
        # n_i . (p_j - c) <= D  =>  -n_i . c - D <= -n_i . p_j
        # -n_i . (p_j - c) <= D =>  n_i . c - D <= n_i . p_j
        
        A_ub = np.zeros((12 * len(pts), 3))
        b_ub = np.zeros(12 * len(pts))
        
        idx = 0
        for nx, ny in normals:
            for px, py in pts:
                proj = px * nx + py * ny
                A_ub[idx, 0] = -nx
                A_ub[idx, 1] = -ny
                A_ub[idx, 2] = -1.0
                b_ub[idx] = -proj
                idx += 1
                A_ub[idx, 0] = nx
                A_ub[idx, 1] = ny
                A_ub[idx, 2] = -1.0
                b_ub[idx] = proj
                idx += 1
        
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
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

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.05):
        """Smooth surrogate for the objective function."""
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
        
        # Log-sum-exp for soft max
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
        
        # 1. Coarse Grid Search
        n_coarse = 60
        for i in range(n_coarse):
            beta = i * (math.pi / 6) / n_coarse
            S, center, D = calc_S_for_beta_fast(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # 2. Fine Grid Search around best
        step = (math.pi / 6) / n_coarse
        for i in range(-5, 6):
            beta = best_beta + i * step / 5
            if beta < 0 or beta >= math.pi/6: continue
            S, center, D = calc_S_for_beta_fast(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # 3. Smooth Surrogate Optimization (Nelder-Mead)
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.05, 0.02]
                start_betas = [best_beta, best_beta + 0.02, best_beta - 0.02]
                
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
                                     options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
                        
                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S_exact, c_exact, _ = calc_S_for_beta_fast(pts, beta_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = c_exact
        except Exception:
            pass
        
        # 4. High-Precision Polish on Exact Objective
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    normals = np.array([
                        (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                         math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
                        for k in range(6)
                    ])
                    pts_centered = pts - np.array([cx, cy])
                    projs = pts_centered @ normals.T
                    max_proj = np.max(np.abs(projs))
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S
                
                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0] + 0.003, best_center[1], best_beta],
                    [best_center[0] - 0.003, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.003, best_beta],
                    [best_center[0], best_center[1] - 0.003, best_beta],
                    [best_center[0], best_center[1], best_beta + 0.003],
                    [best_center[0], best_center[1], best_beta - 0.003],
                ]
                
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
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

    # Sort candidates by a quick estimate to prioritize promising ones
    def quick_estimate(axial_list):
        pts = axial_to_cartesian(axial_list)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        return (max_proj + APOTHEM) / APOTHEM

    estimated_scores = [(quick_estimate(ax), ax) for ax in candidates]
    estimated_scores.sort(key=lambda x: x[0])
    
    # Process candidates with time budget
    n_candidates = len(estimated_scores)
    
    for rank, (est, axial) in enumerate(estimated_scores):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Use a very small safety margin to ensure no overlap
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
