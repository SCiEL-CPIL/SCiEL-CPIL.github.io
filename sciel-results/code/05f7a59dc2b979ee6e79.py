import math
import time
import numpy as np
from scipy.optimize import linprog, minimize_scalar

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    if M == 0:
        return 0.0, 0.0, 0.0

    # Vectorized construction
    v_dot_n = verts @ normals.T  # (M, 6)
    
    # Build A and b efficiently
    # For each normal k: rows are [-nx, -ny, -1] and [nx, ny, -1]
    # b is -v_dot_n[:,k] and v_dot_n[:,k]
    
    A_parts = []
    b_parts = []
    
    for k in range(6):
        nx, ny = normals[k]
        vd = v_dot_n[:, k]
        A_parts.append(np.column_stack([np.full(M, -nx), np.full(M, -ny), np.full(M, -1.0)]))
        b_parts.append(-vd)
        A_parts.append(np.column_stack([np.full(M, nx), np.full(M, ny), np.full(M, -1.0)]))
        b_parts.append(vd)
    
    A = np.vstack(A_parts)
    b = np.concatenate(b_parts)
    
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
            side = R / (_SQRT3/2)
            return cx, cy, side
    except Exception:
        pass
    return None

def _objective_for_angle(verts, angle_deg):
    """Returns the enclosing hex side length for given outer angle, or inf."""
    res = _min_enclosing_hex_side(verts, angle_deg)
    if res is None:
        return float('inf')
    return res[2]

def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    scale = 1.001
    u = (_SQRT3*scale, 0.0)
    v = (_SQRT3/2*scale, 1.5*scale)
    
    if layout_idx == 0:
        # Compact hexagonal: center + 6 ring + 4 from second ring
        target = [(0,0)]
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        target.extend(ring1)
        ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
        target.extend(ring2_sorted[:4])
        
        positions = []
        for ti, tj in target:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x,y))
        return positions[:N]
    elif layout_idx == 1:
        # Row-based: 3 rows
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*_SQRT3*scale + _SQRT3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k*_SQRT3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 3):
            x = k*_SQRT3*scale + _SQRT3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        # 4-3-4 shifted
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*_SQRT3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*_SQRT3*scale + _SQRT3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 2):
            x = k*_SQRT3*scale
            positions.append((x, y3))
        return positions[:N]
    else:
        # Fallback: sorted by distance
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i*u[0] + j*v[0]
                y = i*u[1] + j*v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]

def _refine_angle(verts, angle0):
    """
    Refine the outer angle around angle0 to minimize enclosing hex side length.
    """
    def f(angle_deg):
        return _objective_for_angle(verts, angle_deg)
    
    lo = max(0.0, angle0 - 1.0)
    hi = min(30.0, angle0 + 1.0)
    
    best_angle = angle0
    best_side = f(angle0)
    
    try:
        res_scalar = minimize_scalar(f, bounds=(lo, hi), method='bounded',
                                     options={'xatol': 1e-12, 'maxiter': 500})
        if res_scalar.success:
            candidate_angle = res_scalar.x
            candidate_side = f(candidate_angle)
            if candidate_side < best_side:
                best_angle = candidate_angle
                best_side = candidate_side
    except Exception:
        pass
    
    # Final ultra-tight polish with a few precise LP evaluations
    if time.time() < _DEADLINE - 8.0:
        # Do a very fine local search around best_angle
        fine_step = 0.0005
        sweep_lo = max(0.0, best_angle - 0.005)
        sweep_hi = min(30.0, best_angle + 0.005)
        
        for a in np.arange(sweep_lo, sweep_hi + fine_step/2, fine_step):
            if time.time() > _DEADLINE - 8.0:
                break
            s = _objective_for_angle(verts, a)
            if s < best_side:
                best_side = s
                best_angle = a
    
    # Get the center for the best angle
    res = _min_enclosing_hex_side(verts, best_angle)
    if res is not None:
        cx, cy, side = res
        return best_angle, (cx, cy), side
    else:
        return best_angle, (0.0, 0.0), best_side

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    layouts = [_build_positions(i) for i in range(4)]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Only inner angle 0 is optimal for hexagonal lattice packing
        inner_angle = 0.0
        
        all_vertices = []
        for px, py in positions:
            all_vertices.extend(_hex_vertices(px, py, inner_angle))
        all_vertices = np.array(all_vertices)
        
        # Coarse search: step 2.0 degrees over [0, 30]
        best_coarse_side = float('inf')
        best_coarse_angle = 0.0
        
        for outer_angle_deg in np.arange(0, 30.5, 2.0):
            if time.time() > _DEADLINE - 10.0:
                break
            s = _objective_for_angle(all_vertices, outer_angle_deg)
            if s < best_coarse_side:
                best_coarse_side = s
                best_coarse_angle = outer_angle_deg
        
        # Medium search: step 0.5 degrees around best coarse
        if best_coarse_side < float('inf'):
            start_angle = max(0.0, best_coarse_angle - 2.0)
            end_angle = min(30.0, best_coarse_angle + 2.0)
            
            for outer_angle_deg in np.arange(start_angle, end_angle + 0.25, 0.25):
                if time.time() > _DEADLINE - 10.0:
                    break
                s = _objective_for_angle(all_vertices, outer_angle_deg)
                if s < best_coarse_side:
                    best_coarse_side = s
                    best_coarse_angle = outer_angle_deg
        
        # Fine search: step 0.05 degrees
        if best_coarse_side < float('inf') and time.time() < _DEADLINE - 12.0:
            start_angle = max(0.0, best_coarse_angle - 0.25)
            end_angle = min(30.0, best_coarse_angle + 0.25)
            
            for outer_angle_deg in np.arange(start_angle, end_angle + 0.025, 0.025):
                if time.time() > _DEADLINE - 10.0:
                    break
                s = _objective_for_angle(all_vertices, outer_angle_deg)
                if s < best_coarse_side:
                    best_coarse_side = s
                    best_coarse_angle = outer_angle_deg
        
        # High-precision refinement
        if best_coarse_side < float('inf') and time.time() < _DEADLINE - 15.0:
            refined_angle, refined_center, refined_side = _refine_angle(
                all_vertices, best_coarse_angle)
            if refined_side < best_coarse_side:
                best_coarse_side = refined_side
                best_coarse_angle = refined_angle
                best_center_for_layout = refined_center
            else:
                res = _min_enclosing_hex_side(all_vertices, best_coarse_angle)
                best_center_for_layout = (res[0], res[1]) if res else (0.0, 0.0)
            
            if best_coarse_side < best_side:
                best_side = best_coarse_side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                best_center = best_center_for_layout
                best_outer_angle = best_coarse_angle
    
    # Final global polish on the best configuration
    if best_inner is not None and time.time() < _DEADLINE - 20.0:
        all_vertices = []
        for row in best_inner:
            px, py, ang = row[0], row[1], row[2]
            all_vertices.extend(_hex_vertices(px, py, ang))
        all_vertices = np.array(all_vertices)
        
        refined_angle, refined_center, refined_side = _refine_angle(
            all_vertices, best_outer_angle)
        if refined_side < best_side:
            best_side = refined_side
            best_outer_angle = refined_angle
            best_center = refined_center
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
