import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9

def _time_ok():
    return time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN

def _hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side s, rotated by angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        ang = a + k * math.pi / 3.0
        verts.append((cx + s * math.cos(ang), cy + s * math.sin(ang)))
    return verts

def _support_function(pts, beta):
    """Compute the support function (max projection) of point set pts onto a hexagon with rotation beta.
    Returns the required half-width D such that all points fit inside a hexagon centered at origin."""
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    max_proj = 0.0
    for nx, ny in normals:
        for px, py in pts:
            proj = px * nx + py * ny
            max_proj = max(max_proj, proj, -proj)
    return max_proj

def _support_function_smooth(pts, beta, temperature=0.05):
    """Smooth version using log-sum-exp for the max projection."""
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    all_projs = []
    for nx, ny in normals:
        for px, py in pts:
            proj = px * nx + py * ny
            all_projs.append(proj)
            all_projs.append(-proj)
    
    all_projs = np.array(all_projs)
    max_val = np.max(all_projs)
    if max_val > 100:
        return 1e10
    
    shifted = (all_projs - max_val) / temperature
    log_sum = np.log(np.sum(np.exp(shifted)))
    D = max_val + temperature * log_sum
    return D

def _required_side(pts, beta):
    """Compute the required side length of the outer hexagon."""
    D = _support_function(pts, beta)
    return (D + APOTHEM) / APOTHEM

def _required_side_smooth(pts, beta, temperature=0.05):
    D = _support_function_smooth(pts, beta, temperature)
    return (D + APOTHEM) / APOTHEM

def _axial_to_cartesian(axial_list):
    """Convert axial coordinates to cartesian for unit hexagons on a hexagonal lattice."""
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def _build_candidates():
    """Build candidate axial coordinate sets for 11 hexagons."""
    candidates = []
    
    # Compact 3-4-3-1 arrangement
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)
    
    # 4-3-4 arrangement
    axial_2 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_2)
    
    # 3-5-3 arrangement
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_3)
    
    # 2-4-2-3 arrangement
    axial_4 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_4)
    
    # 4-4-3 arrangement
    axial_5 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_5)
    
    # 5-3-3 arrangement
    axial_6 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_6)
    
    # 3-3-3-2 arrangement
    axial_7 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_7)
    
    # 2-3-2-4 arrangement
    axial_8 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_8)
    
    # 4-2-3-2 arrangement
    axial_9 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_9)
    
    # 2-2-4-3 arrangement
    axial_10 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_10)
    
    # 3-4-4 arrangement
    axial_11 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_11)
    
    # 2-5-4 arrangement
    axial_12 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_12)
    
    # 3-3-5 arrangement
    axial_13 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)
    ]
    candidates.append(axial_13)
    
    # 4-3-4 variant
    axial_14 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_14)
    
    # 2-4-5 arrangement
    axial_15 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)
    ]
    candidates.append(axial_15)
    
    return candidates

def _optimize_rigid(pts, do_polish=True):
    """Optimize the container for a rigid configuration of hexagons."""
    best_S = float('inf')
    best_beta = 0.0
    best_center = None
    
    if not _time_ok():
        return float('inf'), 0.0, None
    
    # Phase 1: Grid search over beta
    for i in range(73):
        if not _time_ok():
            break
        beta = i * (math.pi / 6.0) / 72.0
        S = _required_side(pts, beta)
        if S < best_S:
            best_S = S
            best_beta = beta
            # Find optimal center by symmetry: center of bounding box
            cx = 0.0
            cy = 0.0
            best_center = (cx, cy)
    
    if not do_polish:
        return best_S, best_beta, best_center
    
    # Phase 2: Bounded scalar refinement
    try:
        if _time_ok():
            for b_start in [0.0, math.pi/36, math.pi/18, math.pi/12, math.pi/9, math.pi/7.2]:
                b_end = min(b_start + math.pi/6.0, math.pi/6.0)
                if b_start >= b_end:
                    continue
                res = minimize_scalar(
                    lambda b: _required_side(pts, b),
                    bounds=(b_start, b_end),
                    method='bounded',
                    options={'maxiter': 500, 'xatol': 1e-14}
                )
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    best_center = (0.0, 0.0)
    except Exception:
        pass
    
    # Phase 3: Smooth surrogate with Nelder-Mead for center optimization
    try:
        if _time_ok():
            def smooth_obj(params, temp=0.03):
                cx, cy, beta = params
                beta = max(0.0, min(beta, math.pi / 6.0))
                shifted_pts = np.array([[x - cx, y - cy] for x, y in pts])
                D = _support_function_smooth(shifted_pts, beta, temperature=temp)
                return (D + APOTHEM) / APOTHEM
            
            temperatures = [0.5, 0.2, 0.1, 0.05, 0.02]
            for temp in temperatures:
                if not _time_ok():
                    break
                init = [0.0, 0.0, best_beta]
                res = minimize(
                    lambda p, t=temp: smooth_obj(p, t),
                    init,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
                )
                if res.fun < 1e10:
                    cx_r, cy_r, beta_r = res.x
                    beta_r = max(0.0, min(beta_r, math.pi / 6.0))
                    S_exact = _required_side(np.array([[x - cx_r, y - cy_r] for x, y in pts]), beta_r)
                    if S_exact < best_S:
                        best_S = S_exact
                        best_center = (cx_r, cy_r)
                        best_beta = beta_r
    except Exception:
        pass
    
    # Phase 4: Final high-precision polish
    try:
        if _time_ok():
            def exact_obj(params):
                cx, cy, beta = params
                beta = max(0.0, min(beta, math.pi / 6.0))
                shifted_pts = np.array([[x - cx, y - cy] for x, y in pts])
                D = _support_function(shifted_pts, beta)
                return (D + APOTHEM) / APOTHEM
            
            spatial_offsets = [
                (0, 0), (0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01),
                (0.005, 0.005), (-0.005, -0.005), (0.005, -0.005), (-0.005, 0.005),
                (0.02, 0.02), (-0.02, -0.02),
            ]
            beta_starts = [best_beta, 0.0, math.pi/24, math.pi/12, math.pi/8, math.pi/6]
            
            for ox, oy in spatial_offsets:
                if not _time_ok():
                    break
                for bs in beta_starts:
                    if not _time_ok():
                        break
                    init = [best_center[0] + ox, best_center[1] + oy, bs]
                    res = minimize(
                        exact_obj, init,
                        method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-16, 'fatol': 1e-16, 'adaptive': True}
                    )
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = max(0.0, min(res.x[2], math.pi / 6.0))
    except Exception:
        pass
    
    return best_S, best_beta, best_center

def construct():
    if not _time_ok():
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    candidates = _build_candidates()
    
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = (0.0, 0.0)
    
    # Phase 1: Fast screen all candidates
    screened = []
    for axial in candidates:
        if not _time_ok():
            break
        pts = _axial_to_cartesian(axial)
        # Quick grid screen
        best_S = float('inf')
        best_beta = 0.0
        for i in range(37):
            beta = i * (math.pi / 6.0) / 36.0
            S = _required_side(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
        screened.append((best_S, pts, best_beta))
    
    # Sort and keep top candidates
    screened.sort(key=lambda x: x[0])
    top_k = screened[:15]
    
    # Phase 2: Detailed optimization of top candidates
    for S_screen, pts, beta_screen in top_k:
        if not _time_ok():
            break
        S_opt, beta_opt, center_opt = _optimize_rigid(pts, do_polish=True)
        if S_opt < best_global_S:
            best_global_S = S_opt
            best_global_pts = pts
            best_global_beta = beta_opt
            best_global_center = center_opt
    
    if best_global_pts is None or best_global_S == float('inf'):
        best_global_pts = _axial_to_cartesian(candidates[0])
        best_global_center = (0.0, 0.0)
        best_global_beta = 0.0
        best_global_S = _required_side(best_global_pts, best_global_beta)
    
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Scale points to account for center offset
    pts_final = np.array([[x - cx, y - cy] for x, y in best_global_pts])
    
    # Compute final side length
    D = _support_function(pts_final, best_global_beta)
    S_final = (D + APOTHEM) / APOTHEM
    
    # Add tiny epsilon to ensure validity
    S_final += 1e-10
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x - cx), float(y - cy), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
