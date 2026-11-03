"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin

def _lp_options():
    return {'presolve': True, 'time_limit': 10.0, 'dual_feasibility_tolerance': 1e-12, 'primal_feasibility_tolerance': 1e-12}

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
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            # (vx - cx)*nx + (vy - cy)*ny <= r
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-vx * nx - vy * ny)
            
            # (vx - cx)*nx + (vy - cy)*ny >= -r
            A_ub.append([nx, ny, -1.0])
            b_ub.append(vx * nx + vy * ny)
    
    # r >= 0
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c = np.array([0.0, 0.0, 1.0])
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs', options=_lp_options())
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

def _find_best_outer_lp(inner_hexes):
    """
    Find the best outer hexagon by optimizing the angle using multi-stage search.
    """
    best_side = float('inf')
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    # Stage 1: Coarse sweep (0.1 degree steps)
    angles_coarse = np.arange(0, 30.1, 0.1)
    for angle_deg in angles_coarse:
        if time.time() > _DEADLINE - 10.0:
            break
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
    
    # Stage 2: Fine sweep around best coarse angle (0.01 degree steps)
    search_range = 0.5
    step = 0.01
    fine_angles = np.arange(max(0, best_angle - search_range), min(30, best_angle + search_range) + step, step)
    for angle_deg in fine_angles:
        if time.time() > _DEADLINE - 8.0:
            break
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
    
    # Stage 3: Nelder-Mead refinement
    if time.time() < _DEADLINE - 5.0:
        x0 = np.array([best_angle])
        
        def obj_closure(x):
            if time.time() > _DEADLINE - 3.0:
                return 1e9
            _, _, side, _ = _min_enclosing_hex_side_lp(inner_hexes, float(x[0]))
            return side

        res = minimize(obj_closure, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'adaptive': True})
        
        if res.fun < best_side:
            best_angle = res.x[0]
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
    
    def _generate_positions(scale):
        """Generate hexagonal lattice positions near origin."""
        u = (sqrt3 * scale, 0.0)
        v = (sqrt3 / 2 * scale, 1.5 * scale)
        
        candidates = []
        for i in range(-4, 5):
            for j in range(-4, 5):
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
        return positions
    
    # Use a small gap to ensure disjointness
    scale = 1.001
    positions = _generate_positions(scale)
    
    if not positions:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try inner angles 0 and 30
    for inner_angle in [0.0, 30.0]:
        if time.time() > _DEADLINE - 15.0:
            break
        
        inner = [[pos[0], pos[1], inner_angle] for pos in positions]
        
        cx, cy, side, angle = _find_best_outer_lp(inner)
        
        if side < best_side:
            best_side = side
            best_inner = inner
            best_center = (cx, cy)
            best_angle = angle
    
    # Final high-precision polish
    if best_inner is not None and time.time() < _DEADLINE - 3.0:
        try:
            x0 = np.array([best_angle])
            
            def polish_obj(x):
                if time.time() > _DEADLINE - 2.0:
                    return 1e9
                _, _, side, _ = _min_enclosing_hex_side_lp(best_inner, float(x[0]))
                return side
            
            res_polish = minimize(polish_obj, x0, method='Nelder-Mead',
                                 options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 300, 'adaptive': True})
            
            if res_polish.fun < best_side - 1e-14:
                best_angle = res_polish.x[0]
                cx, cy, side, ang = _min_enclosing_hex_side_lp(best_inner, best_angle)
                best_side = side
                best_center = (cx, cy)
        except Exception:
            pass
    
    if best_inner is None:
        # Fallback if optimization failed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
