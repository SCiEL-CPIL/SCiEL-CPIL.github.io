"""Combined approach: LP-based enclosing hexagon with multiple layouts and high-precision polish."""
import math
import time
import numpy as np
import itertools
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # 145s safety margin within 150s limit
_SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    angles = angle_rad + np.arange(6) * math.pi / 3
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _compute_all_vertices(inner_hexes):
    """Precompute all vertices of inner hexagons as a numpy array (N*6, 2)."""
    n_inner = len(inner_hexes)
    verts = np.zeros((n_inner * 6, 2))
    for idx, (ix, iy, iangle) in enumerate(inner_hexes):
        verts[idx * 6: idx * 6 + 6] = _hex_vertices(ix, iy, iangle)
    return verts

def _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg):
    """For a fixed outer hexagon angle, find optimal center and side length using LP.
    Returns (cx, cy, side) or None on failure.
    """
    n_pts = all_vertices.shape[0]
    if n_pts == 0:
        return (0.0, 0.0, 0.0)
    
    # Normals to the 6 sides of the outer hexagon
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
    
    # For each direction k, constraint: (v - c).n_k <= R for all vertices v
    # => -n_k.c - R <= -v.n_k
    # Also: (c - v).n_k <= R => n_k.c - R <= v.n_k
    
    # Vectorized construction
    v_dot_n = all_vertices @ normals.T  # (n_pts, 6)
    
    A_rows = []
    b_vals = []
    
    for k in range(6):
        nx, ny = normals[k]
        v_dots = v_dot_n[:, k]
        
        # Row 1: -nx*cx - ny*cy - R <= -v_dot
        A_rows.append(np.column_stack([np.full(n_pts, -nx), np.full(n_pts, -ny), np.full(n_pts, -1.0)]))
        b_vals.append(-v_dots)
        
        # Row 2: nx*cx + ny*cy - R <= v_dot
        A_rows.append(np.column_stack([np.full(n_pts, nx), np.full(n_pts, ny), np.full(n_pts, -1.0)]))
        b_vals.append(v_dots)
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    
    # R >= 0
    A = np.vstack([A, [[0.0, 0.0, -1.0]]])
    b = np.append(b, 0.0)
    
    c_obj = [0.0, 0.0, 1.0]
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (_SQRT3 / 2)
            return (cx, cy, side)
    except Exception:
        pass
    
    # Fallback: use centroid and compute required R
    cx_mean = all_vertices[:, 0].mean()
    cy_mean = all_vertices[:, 1].mean()
    centered = all_vertices - np.array([cx_mean, cy_mean])
    projs = centered @ normals.T  # (n_pts, 6)
    R_opt = projs.max()
    if R_opt < 0:
        R_opt = 0.0
    side = R_opt / (_SQRT3 / 2)
    return (cx_mean, cy_mean, side)

def _quick_min_enclosing_hex_side(inner_hexes):
    """Coarse search for minimum enclosing hexagon side."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    all_vertices = _compute_all_vertices(inner_hexes)
    
    # Coarse sweep
    for angle_deg in np.arange(0, 60, 2.0):
        res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Fine sweep
    if best_side < float('inf'):
        for angle_deg in np.arange(max(0.0, best_angle - 5), min(60.0, best_angle + 5), 0.25):
            res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(angle_deg)
    
    return best_center[0], best_center[1], best_side, best_angle

def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons.
    Returns (center_x, center_y, side_length, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    all_vertices = _compute_all_vertices(inner_hexes)
    
    # First pass: coarse sweep
    for angle_deg in np.arange(0, 60, 2.0):
        res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Second pass: fine sweep
    if best_side < float('inf'):
        for angle_deg in np.arange(max(0.0, best_angle - 5), min(60.0, best_angle + 5), 0.25):
            res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(angle_deg)
    
    # Third pass: even finer sweep
    if best_side < float('inf'):
        for angle_deg in np.arange(max(0.0, best_angle - 1.0), min(60.0, best_angle + 1.0), 0.05):
            res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(angle_deg)
    
    # Fourth pass: Nelder-Mead polish
    if best_side < float('inf') and time.time() < _DEADLINE - 15.0:
        from scipy.optimize import minimize
        
        def objective(params):
            ang_p = params[2] % 60.0
            res = _min_enclosing_hex_side_lp(all_vertices, float(ang_p))
            if res:
                return res[2]
            return 1e6
        
        x0 = np.array([best_center[0], best_center[1], best_angle])
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True})
            if res.success:
                ang_nm = float(res.x[2] % 60.0)
                cx_v, cy_v, side_v = _min_enclosing_hex_side_lp(all_vertices, ang_nm)
                if side_v < best_side:
                    best_side = side_v
                    best_center = (cx_v, cy_v)
                    best_angle = ang_nm
        except Exception:
            pass
    
    # Fifth pass: High-precision grid search
    if best_side < float('inf') and time.time() < _DEADLINE - 5.0:
        window = 0.1  # degrees
        steps = 200
        angles_hp = np.linspace(best_angle - window, best_angle + window, steps)
        # Handle wrap-around for angles near 0 or 60
        for ang in angles_hp:
            ang_mod = ang % 60.0
            if ang_mod < 0:
                ang_mod += 60.0
            res = _min_enclosing_hex_side_lp(all_vertices, float(ang_mod))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(ang_mod)
    
    return best_center[0], best_center[1], best_side, best_angle

def _build_positions(layout_idx, scale=1.0000001):
    """Generate 11 positions for given layout index."""
    u = (_SQRT3 * scale, 0.0)
    v = (_SQRT3 / 2 * scale, 1.5 * scale)
    
    if layout_idx == 0:
        # Compact hexagonal: center + 6 ring + 4 from second ring
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
        target.extend(ring2_sorted[:4])
        
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        return positions[:N]
    
    elif layout_idx == 1:
        # Row-based: 3-4-4 arrangement
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k * _SQRT3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 3):
            x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
            positions.append((x, y3))
        return positions[:N]
    
    elif layout_idx == 2:
        # 4-3-4 but shifted
        positions = []
        y1 = -1.5 * scale
        for k in range(-2, 2):
            x = k * _SQRT3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-2, 2):
            x = k * _SQRT3 * scale
            positions.append((x, y3))
        return positions[:N]
    
    elif layout_idx == 3:
        # 2-4-5 arrangement
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 1):
            x = k * _SQRT3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 2):
            x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-2, 3):
            x = k * _SQRT3 * scale
            positions.append((x, y3))
        return positions[:N]
    
    else:
        # Fallback: sorted by distance in triangular lattice
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]

def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 0.0
    
    # Try multiple layouts and inner angles
    layouts = [_build_positions(i) for i in range(5)]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        for inner_angle in [0.0, 30.0]:
            if time.time() > _DEADLINE - 10.0:
                break
            
            inner_candidate = [[pos[0], pos[1], inner_angle] for pos in positions]
            all_vertices = _compute_all_vertices(inner_candidate)
            
            # Coarse search
            best_coarse_side = float('inf')
            best_coarse_angle = 0.0
            best_coarse_center = (0.0, 0.0)
            
            for outer_angle_deg in np.arange(0, 30.1, 1.0):
                if time.time() > _DEADLINE - 10.0:
                    break
                res = _min_enclosing_hex_side_lp(all_vertices, float(outer_angle_deg))
                if res:
                    cx, cy, side = res
                    if side < best_coarse_side:
                        best_coarse_side = side
                        best_coarse_angle = float(outer_angle_deg)
                        best_coarse_center = (cx, cy)
            
            # Fine search
            if best_coarse_side < float('inf'):
                for outer_angle_deg in np.arange(max(0, best_coarse_angle - 1.0),
                                                 min(30.1, best_coarse_angle + 1.0), 0.05):
                    if time.time() > _DEADLINE - 10.0:
                        break
                    res = _min_enclosing_hex_side_lp(all_vertices, float(outer_angle_deg))
                    if res:
                        cx, cy, side = res
                        if side < best_coarse_side:
                            best_coarse_side = side
                            best_coarse_angle = float(outer_angle_deg)
                            best_coarse_center = (cx, cy)
            
            if best_coarse_side < best_side:
                best_side = best_coarse_side
                best_inner = inner_candidate
                best_center = best_coarse_center
                best_outer_angle = best_coarse_angle
                best_inner_angle = inner_angle
    
    # Polish the best configuration
    if best_inner is not None and time.time() < _DEADLINE - 20.0:
        cx, cy, side, angle = _min_enclosing_hex_side(best_inner)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_outer_angle = angle
    
    # Fallback if something went wrong
    if best_inner is None:
        positions = _build_positions(0)
        best_inner = [[pos[0], pos[1], 0.0] for pos in positions]
        cx, cy, best_side, best_outer_angle = _min_enclosing_hex_side(best_inner)
        best_center = (cx, cy)
    
    return best_inner, best_center, best_side, best_outer_angle
