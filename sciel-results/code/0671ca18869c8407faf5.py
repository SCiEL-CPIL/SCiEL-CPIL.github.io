import math
import numpy as np
import time
from scipy.optimize import linprog, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return _get_fallback()

    candidates = []
    
    # Config 1: Plus shape
    c1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(c1)

    # Config 2: Compact cluster
    c2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    candidates.append(c2)

    # Config 3: Pinwheel
    c3 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (1, -2), (-1, -2)
    ]
    candidates.append(c3)
    
    # Config 4: 3x3 block + 2
    c4 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (1, 1), (-1, 1), (0, -1), (1, -1), (-1, -1),
        (2, 0), (0, 2)
    ]
    candidates.append(c4)

    # Config 5: Another
    c5 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, 1), (1, 2), (0, 2)
    ]
    candidates.append(c5)

    best_S = float('inf')
    best_pts = None
    best_beta = 0.0
    best_center = None

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def solve_lp_for_beta(pts, beta):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        n_pts = len(pts)
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for i in range(n_pts):
                px, py = pts[i]
                A_ub.append([nx, ny, -1.0])
                b_ub.append(nx * px + ny * py)
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(nx * px + ny * py))
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, cx, cy, D
            else:
                return float('inf'), None, None, None
        except Exception:
            return float('inf'), None, None, None

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        n_grid = 20
        for i in range(n_grid + 1):
            beta = i * (math.pi / 6) / n_grid
            S, cx, cy, D = solve_lp_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = (cx, cy)
        
        try:
            lo = max(0, best_beta - 0.1)
            hi = min(math.pi/6, best_beta + 0.1)
            
            def obj(beta):
                S, _, _, _ = solve_lp_for_beta(pts, beta)
                return S
            
            res = minimize_scalar(obj, bounds=(lo, hi), method='bounded', options={'maxiter': 100, 'xatol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                _, cx, cy, _ = solve_lp_for_beta(pts, best_beta)
                best_center = (cx, cy)
        except Exception:
            pass
        
        return best_S, best_beta, best_center

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        
        if S < best_S:
            best_S = S
            best_pts = pts
            best_beta = beta
            best_center = center

    if best_center is None:
        return _get_fallback()

    cx, cy = best_center
    angle_deg = math.degrees(best_beta)
    
    normals = np.array([
        (math.cos(best_beta + math.pi / 6.0 + k * math.pi / 3.0),
         math.sin(best_beta + math.pi / 6.0 + k * math.pi / 3.0))
        for k in range(6)
    ])
    pts_centered = best_pts - np.array([cx, cy])
    projs = np.abs(pts_centered @ normals.T)
    D = float(np.max(projs))
    S = (D + APOTHEM) / APOTHEM

    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation

def _get_fallback():
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    pts = np.array(pts)
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    
    max_proj = 0
    for k in range(6):
        ang = math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
    S = (max_proj + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), 0.0])
        
    center = [float(cx), float(cy)]
    side = float(S)
    rotation = 0.0
    return inner, center, side, rotation
