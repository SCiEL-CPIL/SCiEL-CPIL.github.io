"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """
    For a fixed outer hexagon angle, find optimal center and side length using LP.
    Returns (cx, cy, side, angle_deg).
    """
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    n_pts = len(all_vertices)
    if n_pts == 0:
        return 0.0, 0.0, 0.0, angle_deg

    A_ub = []
    b_ub = []
    
    # Outer hexagon normals for given angle
    # The constraint is that all vertices must be inside the outer hexagon.
    # Outer hexagon with inradius r, center (cx, cy), rotation angle_deg.
    # Normal directions are angle_deg + 30 + 60*k.
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            # (vx - cx)*nx + (vy - cy)*ny <= r
            # -nx*cx - ny*cy - r <= -vx*nx - vy*ny
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-vx * nx - vy * ny)
            
            # And the other side:
            # (vx - cx)*nx + (vy - cy)*ny >= -r
            # nx*cx + ny*cy - r <= vx*nx + vy*ny
            A_ub.append([nx, ny, -1.0])
            b_ub.append(vx * nx + vy * ny)
    
    # r >= 0
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c = np.array([0.0, 0.0, 1.0])
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs', options={'presolve': True})
        if result.success:
            cx_opt = result.x[0]
            cy_opt = result.x[1]
            r_opt = result.x[2]
            side = r_opt / (math.sqrt(3) / 2)
            return cx_opt, cy_opt, side, angle_deg
    except Exception:
        pass

    # Fallback: Centroid based
    cx_mean = sum(v[0] for v in all_vertices) / n_pts
    cy_mean = sum(v[1] for v in all_vertices) / n_pts
    r_opt = 0.0
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            proj = (vx - cx_mean) * nx + (vy - cy_mean) * ny
            r_opt = max(r_opt, proj)
            neg_proj = -(vx - cx_mean) * nx - (vy - cy_mean) * ny
            r_opt = max(r_opt, neg_proj)
    side = r_opt / (math.sqrt(3) / 2)
    return cx_mean, cy_mean, side, angle_deg

def _objective(angle_params, inner_hexes):
    """Objective function for Nelder-Mead: minimize side length for a given angle."""
    angle_deg = angle_params[0]
    if time.time() > _DEADLINE - 2.0:
        return 1e9
    try:
        _, _, side, _ = _min_enclosing_hex_side_lp(inner_hexes, angle_deg)
        return side
    except Exception:
        return 1e9

def _find_best_outer_lp(inner_hexes):
    """
    Find the best outer hexagon by optimizing the angle using Nelder-Mead.
    """
    # Initial coarse sweep to find a good starting point
    best_side = float('inf')
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    # Symmetry allows us to search only 0 to 30 degrees
    angles_coarse = np.arange(0, 30.1, 0.5)
    for angle_deg in angles_coarse:
        if time.time() > _DEADLINE - 5.0:
            break
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
            
    # Nelder-Mead refinement on angle
    x0 = np.array([best_angle])
    
    # We use a closure to pass inner_hexes to the objective
    def obj_closure(x):
        return _objective(x, inner_hexes)

    # We restrict the search effectively by starting near the coarse minimum.
    # The landscape is periodic with 60, but symmetric within 0-30.
    res = minimize(obj_closure, x0, method='Nelder-Mead', 
                   options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 300, 'adaptive': True})
    
    # Check if Nelder-Mead improved
    if res.fun < best_side:
        best_angle = res.x[0]
        # Get the final center and side
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, best_angle)
        best_side = side
        best_center = (cx, cy)
    else:
        # Use the best from coarse sweep
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, best_angle)
        best_side = side
        best_center = (cx, cy)

    return best_center[0], best_center[1], best_side, best_angle

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    sqrt3 = math.sqrt(3)
    scale = 1.001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    candidates.sort(key=lambda t: t[0])
    
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == N:
            break
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try inner angles 0 and 30
    for inner_angle in [0.0, 30.0]:
        if time.time() > _DEADLINE - 15.0:
            break
        
        inner = [[pos[0], pos[1], inner_angle] for pos in positions]
        
        if time.time() > _DEADLINE - 10.0:
            # Fast fallback for this configuration if running out of time
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner, 0.0)
            if side < best_side:
                best_side = side
                best_inner = inner
                best_center = (cx, cy)
                best_angle = ang
            continue
            
        cx, cy, side, angle = _find_best_outer_lp(inner)
        
        if side < best_side:
            best_side = side
            best_inner = inner
            best_center = (cx, cy)
            best_angle = angle
    
    if best_inner is None:
        # Fallback if optimization failed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
