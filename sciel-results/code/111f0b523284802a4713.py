import numpy as np
from scipy.optimize import minimize, linprog
import time
import math

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 145.0  # 5s safety margin before 150s kill

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    angle = np.radians(angle_deg)
    angles = angle + np.pi * np.arange(6) / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def axial_to_cartesian(axial_list):
    """Convert axial coordinates to Cartesian for unit hexagon centers in honeycomb."""
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def min_outer_side_for_pts(pts, beta):
    """
    Given fixed point set (inner vertices) and outer hexagon rotation beta,
    find minimum outer hexagon side S and center (cx,cy) via linear programming.
    The 6 supporting lines of the outer hexagon (flat sides) must contain all points.
    For a hexagon with apothem D (inradius), the constraint is:
        |p - c| projected onto each of 6 normal directions <= D
    S = (D + APOTHEM) / APOTHEM  [since inner hex apothem is APOTHEM]
    """
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    
    for nx, ny in normals:
        for px, py in pts:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
    
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
            return S, (cx, cy)
        else:
            return float('inf'), None
    except Exception:
        return float('inf'), None

def objective(params):
    """
    params: [x0,y0,a0, x1,y1,a1, ..., outer_beta]
    Returns minimum outer side length for the given inner configuration and outer rotation.
    """
    n = N
    inner_params = params[:3 * n]
    outer_beta = params[3 * n]
    
    # Compute all inner hexagon vertices
    all_verts = []
    for i in range(n):
        x, y, a = inner_params[3 * i:3 * i + 3]
        v = hex_vertices(x, y, a)
        all_verts.append(v)
    all_verts = np.array(all_verts)  # (n, 6, 2)
    pts = all_verts.reshape(-1, 2)
    
    S, center = min_outer_side_for_pts(pts, outer_beta)
    return S

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_result = None
    best_score = float('inf')
    
    # Generate candidate axial layouts for 11 hexagons
    candidates = []
    
    # 1: Center + 6 ring + 4 extensions
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ])
    
    # 2: 3x3 grid + 2 extras
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (3, 1), (-1, 1)
    ])
    
    # 3: 4-4-3 staggered
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ])
    
    # 4: 4-4-3 different stagger
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2), (2, 2)
    ])
    
    # 5: 5-3-3
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2)
    ])
    
    # 6: 3-5-3
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2), (2, 2)
    ])
    
    # 7: Compact star
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ])
    
    # 8: 2-4-3-2
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ])
    
    # 9: 2-5-2-2
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3)
    ])
    
    # 10: 3-4-3-1
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ])
    
    # 11: 1-4-4-2
    candidates.append([
        (0, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ])
    
    # 12: Diagonal-ish
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (2, 2), (3, 2),
        (3, 3)
    ])
    
    # 13: 2-2-2-2-3
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4), (2, 4)
    ])
    
    # 14: 3-3-3-2
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ])
    
    # 15: Cross shape
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ])
    
    # 16: 4-3-4
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ])
    
    # 17: 5-2-4
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ])
    
    # 18: 3-4-2-2
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3)
    ])
    
    # 19: 2-3-4-2
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ])
    
    # 20: 4-2-3-2
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ])
    
    # 21: 2-4-3-2
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ])
    
    # 22: 3-2-4-2
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ])
    
    # 23: 2-3-3-3
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ])
    
    # 24: 3-3-3-2
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ])
    
    # 25: 4-1-4-2
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ])
    
    # 26: 2-1-4-4
    candidates.append([
        (0, 0), (1, 0),
        (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ])
    
    # 27: 1-2-4-4
    candidates.append([
        (0, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ])
    
    # 28: 2-2-4-3
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3), (2, 3)
    ])
    
    # 29: 3-2-4-2
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ])
    
    # 30: 4-2-3-2
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ])
    
    # 31: 2-4-2-3
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3)
    ])
    
    # 32: 3-4-1-3
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2),
        (0, 3), (1, 3), (2, 3)
    ])
    
    # 33: 2-5-1-3
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (2, 2),
        (0, 3), (1, 3), (2, 3)
    ])
    
    # 34: 4-3-1-3
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2),
        (0, 3), (1, 3), (2, 3)
    ])
    
    # 35: 3-5-1-2
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (2, 2),
        (0, 3), (1, 3)
    ])
    
    # 36: 5-4-2
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2)
    ])
    
    # 37: 5-3-3
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2)
    ])
    
    # 38: 4-4-3
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ])
    
    # 39: 3-4-4
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ])
    
    # 40: 2-4-5
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)
    ])
    
    # 41: 5-4-2
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2)
    ])
    
    # 42: 2-5-4
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ])
    
    # 43: 4-5-2
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2)
    ])
    
    # 44: 2-2-7
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2)
    ])
    
    # 45: 7-2-2
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2)
    ])
    
    # 46: 3-7-1
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
        (0, 2)
    ])
    
    # 47: 1-7-3
    candidates.append([
        (0, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
        (0, 2), (1, 2), (2, 2)
    ])
    
    # 48: 2-7-2
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
        (0, 2), (1, 2)
    ])
    
    # 49: 6-3-2
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2)
    ])
    
    # 50: 2-3-6
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)
    ])
    
    # Now optimize each candidate
    for ci, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        pts_init = axial_to_cartesian(axial)
        
        # Build initial params: all hexagons at 0 rotation, positions from axial
        params0 = []
        for x, y in pts_init:
            params0.extend([x, y, 0.0])
        params0.append(0.0)  # outer beta
        
        # Try a few outer beta starting points
        for beta_start in [0.0, math.pi / 12, math.pi / 6, math.pi / 18, math.pi / 9]:
            if time.time() - START_TIME > TIME_LIMIT:
                break
            
            params0[-1] = beta_start
            score0 = objective(params0)
            
            try:
                res = minimize(objective, params0, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x.copy()
            except Exception:
                pass
        
        # Also try with some hexagons rotated
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Try rotating alternating hexagons by 30 degrees
        params_rot = []
        for i, (x, y) in enumerate(pts_init):
            ang = 30.0 if i % 2 == 0 else 0.0
            params_rot.extend([x, y, ang])
        params_rot.append(0.0)
        
        for beta_start in [0.0, math.pi / 12, math.pi / 6]:
            if time.time() - START_TIME > TIME_LIMIT:
                break
            
            params_rot[-1] = beta_start
            try:
                res = minimize(objective, params_rot, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x.copy()
            except Exception:
                pass
    
    if best_result is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Final polish
    if time.time() - START_TIME < TIME_LIMIT - 10:
        try:
            res = minimize(objective, best_result, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Parse best result
    inner = []
    for i in range(N):
        x, y, a = best_result[3 * i:3 * i + 3]
        inner.append([float(x), float(y), float(a)])
    
    # Compute outer center and side
    all_verts = []
    for i in range(N):
        x, y, a = best_result[3 * i:3 * i + 3]
        v = hex_vertices(x, y, a)
        all_verts.append(v)
    all_verts = np.array(all_verts)
    pts = all_verts.reshape(-1, 2)
    
    outer_beta = best_result[3 * N]
    S, center = min_outer_side_for_pts(pts, outer_beta)
    
    if center is None or not np.isfinite(S):
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return inner, (float(center[0]), float(center[1])), float(S), float(outer_beta)
