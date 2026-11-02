import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Correct non-overlapping hexagonal grid: 1 center + 6 ring1 + 4 ring2
    # Using axial coordinates (q, r) where Cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    axial_coords = [
        (0, 0),      # center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # ring 1
        (2, 0), (0, 2), (-2, 2), (-2, 0)  # ring 2 (4 of 6)
    ]
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)
    
    pts = axial_to_cartesian(axial_coords)
    
    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        # Vectorized normal computation
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
        
        # Build LP matrices efficiently
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
        
        # Coarse grid search over beta
        for i in range(73):
            beta = i * (math.pi / 6) / 72
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta with bounded scalar optimization
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
        
        # Smooth surrogate optimization with adaptive temperature
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
        
        # Final high-precision polish using exact objective
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
                    [best_center[0] + 0.01, best_center[1] + 0.01, best_beta],
                    [best_center[0] - 0.01, best_center[1] - 0.01, best_beta],
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

    best_S, best_beta, best_center = optimize_shape(pts)
    
    if best_center is None or best_S == float('inf'):
        best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_S = (max_proj + APOTHEM) / APOTHEM
    
    # Second pass: try to improve with more aggressive optimization
    if best_S < float('inf') and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        def perturbed_objective(params, base_pts, base_beta, base_center):
            cx, cy, beta = params
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            
            centered = base_pts - np.array([cx, cy])
            projs = np.abs(centered @ normals.T)
            max_proj = np.max(projs)
            
            S = (max_proj + APOTHEM) / APOTHEM
            return S
        
        np.random.seed(42)
        for trial in range(5):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            dx = np.random.uniform(-0.1, 0.1)
            dy = np.random.uniform(-0.1, 0.1)
            dbeta = np.random.uniform(-0.1, 0.1)
            
            init = [best_center[0] + dx, best_center[1] + dy, 
                   ((best_beta + dbeta) % (math.pi/6) + math.pi/6) % (math.pi/6)]
            
            res = minimize(perturbed_objective, init, args=(pts, best_beta, best_center),
                         method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-13})
            
            if res.fun < best_S:
                best_S = res.fun
                best_center = (res.x[0], res.x[1])
                best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)

    cx, cy = best_center
    angle_deg = math.degrees(best_beta)
    
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    max_proj = 0
    for k in range(6):
        ang = best_beta + math.pi / 6.0 + k * math.pi / 3.0
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
