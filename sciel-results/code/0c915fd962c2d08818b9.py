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

def _hexagon_vertices(x, y, angle, side=1.0):
    """Return the 6 vertices of a regular hexagon with center (x,y), side=side, rotation=angle."""
    angles = angle + np.arange(6) * math.pi / 3.0
    r = side
    return np.array([[x + r * math.cos(a), y + r * math.sin(a)] for a in angles])

def _min_enclosing_hexagon_side(pts, beta):
    """
    Compute the minimum enclosing regular hexagon side length for point set pts,
    with hexagon orientation beta (rotation of the hexagon's first normal).
    Returns (side, center_x, center_y).
    """
    angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    n_pts = len(pts)
    n_constraints = 12 * n_pts
    A_ub = np.zeros((n_constraints, 3))
    b_ub = np.zeros(n_constraints)
    
    for k in range(6):
        nx, ny = normals[k]
        projs = pts[:, 0] * nx + pts[:, 1] * ny
        start = 2 * k * n_pts
        A_ub[start:start+n_pts, 0] = -nx
        A_ub[start:start+n_pts, 1] = -ny
        A_ub[start:start+n_pts, 2] = -1.0
        b_ub[start:start+n_pts] = -projs
        A_ub[start+n_pts:start+2*n_pts, 0] = nx
        A_ub[start+n_pts:start+2*n_pts, 1] = ny
        A_ub[start+n_pts:start+2*n_pts, 2] = -1.0
        b_ub[start+n_pts:start+2*n_pts] = projs
    
    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            D = res.x[2]
            cx = res.x[0]
            cy = res.x[1]
            S = (D + APOTHEM) / APOTHEM
            return S, cx, cy
        else:
            return float('inf'), 0.0, 0.0
    except Exception:
        return float('inf'), 0.0, 0.0

def _exact_S(pts, cx, cy, beta):
    """Exact computation of S for given center and rotation."""
    angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    pts_centered = pts - np.array([cx, cy])
    projs = pts_centered @ normals.T
    D = np.max(np.abs(projs))
    return (D + APOTHEM) / APOTHEM

def _smooth_S(pts, cx, cy, beta, temperature=0.05):
    """Smooth surrogate using log-sum-exp."""
    angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    pts_centered = pts - np.array([cx, cy])
    projs = pts_centered @ normals.T
    all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
    max_val = np.max(all_projs)
    if max_val > 200:
        return float('inf')
    shifted = (all_projs - max_val) / temperature
    shifted = np.clip(shifted, -50, 50)
    log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
    D = temperature * log_sum
    return (D + APOTHEM) / APOTHEM

def _optimize_orientation(pts):
    """1D scalar minimization over outer hexagon orientation beta in [0, pi/6]."""
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return float('inf'), 0.0, None
    
    best_S = float('inf')
    best_beta = 0.0
    best_center = None
    
    # Dense grid search
    n_grid = 97
    for i in range(n_grid):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        beta = i * (math.pi / 6) / (n_grid - 1)
        S, cx, cy = _min_enclosing_hexagon_side(pts, beta)
        if S < best_S:
            best_S = S
            best_beta = beta
            best_center = (cx, cy)
    
    if best_center is None:
        return float('inf'), 0.0, None
    
    # Golden-section refinement
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            window = (math.pi / 6) / (n_grid - 1)
            lo = max(0.0, best_beta - 2 * window)
            hi = min(math.pi / 6, best_beta + 2 * window)
            
            def obj_1d(b):
                S, _, _ = _min_enclosing_hexagon_side(pts, b)
                return S
            
            res = minimize_scalar(obj_1d, bounds=(lo, hi), method='bounded',
                                 options={'maxiter': 200, 'xatol': 1e-12})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                _, cx, cy = _min_enclosing_hexagon_side(pts, best_beta)
                best_center = (cx, cy)
    except Exception:
        pass
    
    return best_S, best_beta, best_center

def _optimize_joint(pts):
    """Joint optimization of (cx, cy, beta) using smooth surrogate then exact polish."""
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return float('inf'), 0.0, None
    
    best_S = float('inf')
    best_beta = 0.0
    best_center = None
    
    # Stage 1: Fast 1D orientation-only optimization
    S1, beta1, center1 = _optimize_orientation(pts)
    if center1 is not None:
        best_S = S1
        best_beta = beta1
        best_center = center1
    
    if best_center is None:
        return float('inf'), 0.0, None
    
    # Stage 2: Joint (cx, cy, beta) optimization with smooth surrogate
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            temperatures = [0.05, 0.02, 0.01]
            for temp in temperatures:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                
                def smooth_obj(params, t=temp):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return _smooth_S(pts, cx, cy, beta, temperature=t)
                
                start_betas = [best_beta, best_beta + 0.03, best_beta - 0.03]
                for sb in start_betas:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    init = [best_center[0], best_center[1], sb]
                    try:
                        res = minimize(smooth_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-11})
                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                            S_exact = _exact_S(pts, cx_r, cy_r, beta_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = (cx_r, cy_r)
                    except Exception:
                        pass
    except Exception:
        pass
    
    # Stage 3: High-precision polish with exact objective
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def exact_obj(params):
                cx, cy, beta = params
                beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                return _exact_S(pts, cx, cy, beta)
            
            polish_starts = [
                [best_center[0], best_center[1], best_beta],
                [best_center[0], best_center[1], best_beta + 0.005],
                [best_center[0], best_center[1], best_beta - 0.005],
                [best_center[0] + 0.005, best_center[1], best_beta],
                [best_center[0] - 0.005, best_center[1], best_beta],
                [best_center[0], best_center[1] + 0.005, best_beta],
                [best_center[0], best_center[1] - 0.005, best_beta],
            ]
            
            for init in polish_starts:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                try:
                    res = minimize(exact_obj, init, method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-13})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                except Exception:
                    pass
    except Exception:
        pass
    
    return best_S, best_beta, best_center

def _axial_to_cartesian(axial_list):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def _generate_candidate_configs():
    """Generate a diverse set of candidate hexagon cluster configurations."""
    configs = []
    
    # Standard hexagonal cluster
    configs.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    # Cross
    configs.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # 3x3 block + 2
    configs.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    # 4x3 minus corners
    configs.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    # Staircase
    configs.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    # Compact 3x3+2
    configs.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    # Star
    configs.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    # Elongated
    configs.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    # Asymmetric star
    configs.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    # Compact with top
    configs.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    # Wide base
    configs.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    # Tall narrow
    configs.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)])
    # Star with asymmetric outer
    configs.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)])
    # Star with diagonal
    configs.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    # Dense cluster
    configs.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    # Zigzag
    configs.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])
    # Star with top cluster
    configs.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)])
    # Trapezoid
    configs.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    # Wide top
    configs.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)])
    # Star with inner diagonal
    configs.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)])
    # 4x3 variant
    configs.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    # Wide with top
    configs.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    # Symmetric star
    configs.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # Very tall
    configs.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)])
    # 3x3+row
    configs.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    # Dense 4-wide
    configs.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    # Star with asymmetric bottom
    configs.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)])
    # Tall with top row
    configs.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    # Additional
    configs.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    configs.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)])
    configs.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    configs.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    configs.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    configs.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    configs.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)])
    
    return configs

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    configs = _generate_candidate_configs()
    
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None
    
    for idx, axial in enumerate(configs):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = _axial_to_cartesian(axial)
        S, beta, center = _optimize_joint(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center
    
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = _axial_to_cartesian(configs[0])
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
    
    # Final: compute side length with tiny safety scale
    SCALE = 1.0 + 1e-14
    pts_scaled = best_global_pts.copy()
    for i in range(len(pts_scaled)):
        pts_scaled[i, 0] = cx + (pts_scaled[i, 0] - cx) * SCALE
        pts_scaled[i, 1] = cy + (pts_scaled[i, 1] - cy) * SCALE
    
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
