<<<<<<< SEARCH
import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
=======
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
>>>>>>> REPLACE
<<<<<<< SEARCH
def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
    # Normals to the 6 sides of the outer hexagon
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    if M == 0:
        return 0.0, 0.0, 0.0

    v_dot_n = verts @ normals.T
    
    A_rows = []
    b_vals = []
    
    for k in range(6):
        nx, ny = normals[k]
        v_dots = v_dot_n[:, k]
        
        A_rows.append(np.column_stack([np.full(M, -nx), np.full(M, -ny), np.full(M, -1.0)]))
        b_vals.append(-v_dots)
        
        A_rows.append(np.column_stack([np.full(M, nx), np.full(M, ny), np.full(M, -1.0)]))
        b_vals.append(v_dots)
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    
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
=======
def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
    # Normals to the 6 sides of the outer hexagon
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    if M == 0:
        return 0.0, 0.0, 0.0

    v_dot_n = verts @ normals.T
    
    A_rows = []
    b_vals = []
    
    for k in range(6):
        nx, ny = normals[k]
        v_dots = v_dot_n[:, k]
        
        A_rows.append(np.column_stack([np.full(M, -nx), np.full(M, -ny), np.full(M, -1.0)]))
        b_vals.append(-v_dots)
        
        A_rows.append(np.column_stack([np.full(M, nx), np.full(M, ny), np.full(M, -1.0)]))
        b_vals.append(v_dots)
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    
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

def _compute_outer_side(positions, inner_angles, outer_angle_deg):
    """
    Compute the minimum outer hexagon side length for given positions and angles.
    Uses log-sum-exp soft-max surrogate for smooth optimization.
    """
    all_vertices = []
    for i, (px, py) in enumerate(positions):
        ang = inner_angles[i] if isinstance(inner_angles, (list, tuple, np.ndarray)) else inner_angles
        all_vertices.extend(_hex_vertices(px, py, ang))
    all_vertices = np.array(all_vertices)
    
    if len(all_vertices) == 0:
        return 0.0
    
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    v_dot_n = all_vertices @ normals.T
    max_dots = np.max(v_dot_n, axis=0)
    min_dots = np.min(v_dot_n, axis=0)
    
    # Soft-max with temperature for smoothness
    beta = 50.0
    soft_max = np.zeros(6)
    soft_min = np.zeros(6)
    for k in range(6):
        vals_max = v_dot_n[:, k]
        vals_min = v_dot_n[:, k]
        # Use log-sum-exp for soft max
        m_max = np.max(vals_max)
        soft_max[k] = m_max + np.log(np.sum(np.exp((vals_max - m_max) * beta)) / beta) / beta
        # Soft min is -soft_max of negatives
        m_min = np.min(vals_min)
        soft_min[k] = m_min - np.log(np.sum(np.exp((vals_min - m_min) * beta)) / beta) / beta
    
    # The required radius R must satisfy: for each normal direction k,
    # R >= (max_dot - min_dot) / 2 approximately, but more precisely:
    # The hexagon with center (cx,cy) and circumradius R contains all points iff
    # for each normal n_k: max((p-cx-cy)·n_k) <= R * (sqrt(3)/2) and similar
    # Actually, the apothem is R*sqrt(3)/2, so we need max dot product <= R*sqrt(3)/2
    
    # For a hexagon centered at origin with circumradius R, the constraint is:
    # max over vertices of (v · n_k) <= R * sqrt(3)/2 for each of the 6 directions
    # So R >= max_k (soft_max[k] - soft_min[k]) / (2 * sqrt(3)/2) = max_k (...) / sqrt(3)
    
    # Wait, let me reconsider. The hexagon has 6 sides. The distance from center to each side is the apothem = R * cos(30°) = R * sqrt(3)/2.
    # So we need: for each side direction k, the max projection of vertices onto normal k minus center projection <= apothem
    # And min projection minus center projection >= -apothem
    
    # If we center the hexagon optimally, the center should be at the midpoint of the range for each direction.
    # So for direction k: center_k = (max_k + min_k) / 2, and we need (max_k - min_k)/2 <= R * sqrt(3)/2
    # Thus R >= max_k (max_k - min_k) / sqrt(3)
    
    ranges = soft_max - soft_min
    R = np.max(ranges) / _SQRT3
    side = R / (_SQRT3/2)
    
    return side
>>>>>>> REPLACE
<<<<<<< SEARCH
def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    scale = 1.0000001
    u = (_SQRT3*scale, 0.0)
    v = (_SQRT3/2*scale, 1.5*scale)
    
    if layout_idx == 0:
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
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i*u[0] + j*v[0]
                y = i*u[1] + j*v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]
=======
def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    scale = 1.0000001
    u = (_SQRT3*scale, 0.0)
    v = (_SQRT3/2*scale, 1.5*scale)
    
    if layout_idx == 0:
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
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i*u[0] + j*v[0]
                y = i*u[1] + j*v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]

def _smooth_outer_side(params, positions_init, outer_angle_deg):
    """
    Smooth surrogate for outer hexagon side length using log-sum-exp.
    params: [x1,y1,...,xN,yN, angle1,...,angleN]
    Returns the soft-max-based outer side length.
    """
    N_local = len(positions_init)
    positions = []
    inner_angles = []
    for i in range(N_local):
        x = params[2*i]
        y = params[2*i+1]
        positions.append((x, y))
        inner_angles.append(params[2*N_local + i])
    
    return _compute_outer_side(positions, inner_angles, outer_angle_deg)
>>>>>>> REPLACE
<<<<<<< SEARCH
def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 0.0
    
    layouts = [_build_positions(i) for i in range(4)]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        for inner_angle in [0.0, 30.0]:
            if time.time() > _DEADLINE - 10.0:
                break
            
            all_vertices = []
            for px, py in positions:
                all_vertices.extend(_hex_vertices(px, py, inner_angle))
            all_vertices = np.array(all_vertices)
            
            best_coarse_side = float('inf')
            best_coarse_angle = 0.0
            best_coarse_center = (0.0, 0.0)
            
            for outer_angle_deg in np.arange(0, 30.1, 1.0):
                if time.time() > _DEADLINE - 10.0:
                    break
                res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_coarse_side:
                        best_coarse_side = side
                        best_coarse_angle = outer_angle_deg
                        best_coarse_center = (cx, cy)
            
            if best_coarse_angle > 0 or best_coarse_side < float('inf'):
                start_angle = max(0.0, best_coarse_angle - 1.0)
                end_angle = min(30.0, best_coarse_angle + 1.0)
                
                for outer_angle_deg in np.arange(start_angle, end_angle + 0.05, 0.05):
                    if time.time() > _DEADLINE - 10.0:
                        break
                    res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                    if res:
                        cx, cy, side = res
                        if side < best_coarse_side:
                            best_coarse_side = side
                            best_coarse_angle = outer_angle_deg
                            best_coarse_center = (cx, cy)
                
                if time.time() < _DEADLINE - 15.0 and best_coarse_side < float('inf'):
                    start_angle = max(0.0, best_coarse_angle - 0.05)
                    end_angle = min(30.0, best_coarse_angle + 0.05)
                    for outer_angle_deg in np.arange(start_angle, end_angle + 0.005, 0.005):
                        if time.time() > _DEADLINE - 10.0:
                            break
                        res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                        if res:
                            cx, cy, side = res
                            if side < best_coarse_side:
                                best_coarse_side = side
                                best_coarse_angle = outer_angle_deg
                                best_coarse_center = (cx, cy)
            
            if best_coarse_side < best_side:
                best_side = best_coarse_side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                best_center = best_coarse_center
                best_outer_angle = best_coarse_angle
                best_inner_angle = inner_angle
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
=======
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
        if time.time() > _DEADLINE - 15.0:
            break
        
        # First, do the original coarse-to-fine outer angle search
        for inner_angle in [0.0, 30.0]:
            if time.time() > _DEADLINE - 15.0:
                break
            
            all_vertices = []
            for px, py in positions:
                all_vertices.extend(_hex_vertices(px, py, inner_angle))
            all_vertices = np.array(all_vertices)
            
            best_coarse_side = float('inf')
            best_coarse_angle = 0.0
            best_coarse_center = (0.0, 0.0)
            
            for outer_angle_deg in np.arange(0, 30.1, 1.0):
                if time.time() > _DEADLINE - 15.0:
                    break
                res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_coarse_side:
                        best_coarse_side = side
                        best_coarse_angle = outer_angle_deg
                        best_coarse_center = (cx, cy)
            
            if best_coarse_angle > 0 or best_coarse_side < float('inf'):
                start_angle = max(0.0, best_coarse_angle - 1.0)
                end_angle = min(30.0, best_coarse_angle + 1.0)
                
                for outer_angle_deg in np.arange(start_angle, end_angle + 0.05, 0.05):
                    if time.time() > _DEADLINE - 15.0:
                        break
                    res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                    if res:
                        cx, cy, side = res
                        if side < best_coarse_side:
                            best_coarse_side = side
                            best_coarse_angle = outer_angle_deg
                            best_coarse_center = (cx, cy)
                
                if time.time() < _DEADLINE - 20.0 and best_coarse_side < float('inf'):
                    start_angle = max(0.0, best_coarse_angle - 0.05)
                    end_angle = min(30.0, best_coarse_angle + 0.05)
                    for outer_angle_deg in np.arange(start_angle, end_angle + 0.005, 0.005):
                        if time.time() > _DEADLINE - 15.0:
                            break
                        res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                        if res:
                            cx, cy, side = res
                            if side < best_coarse_side:
                                best_coarse_side = side
                                best_coarse_angle = outer_angle_deg
                                best_coarse_center = (cx, cy)
            
            if best_coarse_side < best_side:
                best_side = best_coarse_side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                best_center = best_coarse_center
                best_outer_angle = best_coarse_angle
        
        # Now try smooth optimization to refine positions and angles
        if time.time() > _DEADLINE - 25.0:
            break
        
        # Try to improve using L-BFGS-B on the smooth surrogate
        for outer_angle_deg in [best_outer_angle if best_outer_angle > 0 else 0.0, 15.0, 30.0]:
            if time.time() > _DEADLINE - 20.0:
                break
            
            # Initialize params from current best positions
            init_params = []
            for px, py in positions:
                init_params.append(px)
                init_params.append(py)
            for px, py in positions:
                init_params.append(0.0)  # initial inner angles
            
            def objective(params):
                if time.time() > _DEADLINE - 18.0:
                    return float('inf')
                return _smooth_outer_side(params, positions, outer_angle_deg)
            
            try:
                res = minimize(objective, init_params, method='L-BFGS-B',
                             options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.success and res.fun < best_side:
                    # Verify with exact LP
                    refined_positions = []
                    refined_angles = []
                    for i in range(N):
                        refined_positions.append((res.x[2*i], res.x[2*i+1]))
                        refined_angles.append(res.x[2*N + i])
                    
                    all_vertices = []
                    for i, (px, py) in enumerate(refined_positions):
                        all_vertices.extend(_hex_vertices(px, py, refined_angles[i]))
                    all_vertices = np.array(all_vertices)
                    
                    res_exact = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                    if res_exact:
                        cx, cy, side = res_exact
                        if side < best_side:
                            best_side = side
                            best_inner = [[refined_positions[i][0], refined_positions[i][1], refined_angles[i]] for i in range(N)]
                            best_center = (cx, cy)
                            best_outer_angle = outer_angle_deg
            except Exception:
                pass
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
>>>>>>> REPLACE
