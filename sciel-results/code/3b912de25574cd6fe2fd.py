import math
import numpy as np
import time
from scipy.optimize import linprog, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets (axial coordinates)
    # These are the centers of the 11 unit hexagons.
    # The coordinates are in a system where distance between adjacent centers is 1.
    # Wait, axial coords (q, r) map to Cartesian:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # Distance between (0,0) and (1,0) is sqrt(3).
    # This is the distance between centers of touching hexagons.
    # So these are valid non-overlapping configurations.
    
    candidates = []

    # 1. 7-center flower + 4 arms
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_1)

    # 2. 3x3 block + 2
    axial_2 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_2)

    # 3. 4x2 block + 3
    axial_3 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_3)

    # 4. Star
    axial_4 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_4)

    # 5. Compact 3x3 + 2 diagonal
    axial_5 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    # 6. Wide base
    axial_6 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_6)

    # 7. Tall
    axial_7 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_7)

    # 8. Asymmetric
    axial_8 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_8)

    # 9. 4x3 minus corners
    axial_9 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_9)

    # 10. Zigzag
    axial_10 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_10)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def fast_calc_S(pts, beta):
        """Calculate minimum side length S_eff for given beta using width method."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Project points onto normals
        projs = pts @ normals.T  # (n_pts, 6)
        
        # Width in each direction
        widths = np.max(projs, axis=0) - np.min(projs, axis=0)
        
        max_width = np.max(widths)
        S_eff = max_width / math.sqrt(3)
        return S_eff

    def get_center_and_S(pts, beta):
        """Use LP to find precise center and confirm S."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        n_pts = len(pts)
        A_ub = np.zeros((12 * n_pts, 3))
        b_ub = np.zeros(12 * n_pts)
        
        for k in range(6):
            nx, ny = normals[k]
            base = 2 * k * n_pts
            A_ub[base:base+n_pts, 0] = nx
            A_ub[base:base+n_pts, 1] = ny
            A_ub[base:base+n_pts, 2] = -1.0
            b_ub[base:base+n_pts] = pts[:, 0] * nx + pts[:, 1] * ny
            
            A_ub[base+n_pts:base+2*n_pts, 0] = -nx
            A_ub[base+n_pts:base+2*n_pts, 1] = -ny
            A_ub[base+n_pts:base+2*n_pts, 2] = -1.0
            b_ub[base+n_pts:base+2*n_pts] = -(pts[:, 0] * nx + pts[:, 1] * ny)
            
        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = max(res.x[2], 0.0)
                cx = res.x[0]
                cy = res.x[1]
                S_eff = D / APOTHEM
                return S_eff, (cx, cy)
            else:
                return float('inf'), None
        except Exception:
            return float('inf'), None

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        
        # Stage 1: Grid search for beta
        best_beta = 0.0
        best_S_eff = float('inf')
        n_grid = 50
        for i in range(n_grid):
            beta = i * (math.pi / 6) / (n_grid - 1)
            S_eff = fast_calc_S(pts, beta)
            if S_eff < best_S_eff:
                best_S_eff = S_eff
                best_beta = beta
                
        # Stage 2: Refine beta
        try:
            res = minimize_scalar(lambda b: fast_calc_S(pts, b), 
                                  bounds=(max(0, best_beta - math.pi/12), min(math.pi/6, best_beta + math.pi/12)), 
                                  method='bounded',
                                  options={'maxiter': 500, 'xatol': 1e-12})
            if res.fun < best_S_eff:
                best_S_eff = res.fun
                best_beta = res.x
        except Exception:
            pass
        
        # Stage 3: Get precise center and S using LP
        S_eff_precise, center = get_center_and_S(pts, best_beta)
        if S_eff_precise < best_global_S:
            best_global_S = S_eff_precise
            best_global_pts = pts
            best_global_beta = best_beta
            best_global_center = center

    if best_global_center is None:
        # Fallback
        pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_global_beta = 0.0
        S_eff = fast_calc_S(pts, best_global_beta)
        best_global_S = S_eff
        best_global_pts = pts

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # The reported side length is S_eff + 1
    side = best_global_S + 1.0
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
