# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _check_separation_fast(positions):
    """
    Check if all pairs of unit hexagons at given positions (angle 0) are separated.
    Returns True if separated, False otherwise.
    Uses Separating Axis Theorem.
    """
    n = len(positions)
    base_verts = _hex_vertices(0, 0, 1.0, 0)
    
    edges = np.empty((6, 2))
    for k in range(6):
        edges[k] = base_verts[(k + 1) % 6] - base_verts[k]
    normals = np.empty((6, 2))
    for k in range(6):
        normals[k] = np.array([-edges[k][1], edges[k][0]])
    
    base_dots = base_verts @ normals.T  # (6, 6)
    center_dots = positions @ normals.T  # (n, 6)
    
    base_min = base_dots.min(axis=0)  # (6,)
    base_max = base_dots.max(axis=0)  # (6,)
    
    for i in range(n):
        for j in range(i + 1, n):
            proj_i_min = center_dots[i] + base_min
            proj_i_max = center_dots[i] + base_max
            proj_j_min = center_dots[j] + base_min
            proj_j_max = center_dots[j] + base_max
            
            sep1 = proj_i_max < proj_j_min - 1e-6
            sep2 = proj_j_max < proj_i_min - 1e-6
            if np.any(sep1) or np.any(sep2):
                continue
            else:
                return False
    return True


def _compute_s_lp(positions, angle_deg):
    """
    Use linear programming to find the optimal outer center and side length.
    Returns (cx, cy, s).
    """
    all_verts = np.empty((72, 2))
    base_verts = _hex_vertices(0, 0, 1.0, 0)
    for i in range(12):
        all_verts[i * 6:(i + 1) * 6] = positions[i] + base_verts
    
    a_rad = np.deg2rad(angle_deg)
    normal_angles = a_rad + np.deg2rad(30 + 60 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    n_verts = len(all_verts)
    n_normals = 6
    
    c_matrix = np.zeros((n_verts * n_normals, 3))
    b_vec = np.zeros(n_verts * n_normals)
    
    for i, v in enumerate(all_verts):
        for k in range(n_normals):
            row = i * n_normals + k
            c_matrix[row, 0] = -normals[k, 0]
            c_matrix[row, 1] = -normals[k, 1]
            c_matrix[row, 2] = -1.0
            b_vec[row] = -np.dot(v, normals[k])
    
    c_obj = np.array([0.0, 0.0, 1.0])
    
    try:
        res = linprog(c_obj, A_ub=c_matrix, b_ub=b_vec, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if res.success:
            cx, cy, t = res.x
            s = t * 2.0 / np.sqrt(3.0)
            return cx, cy, s
    except Exception:
        pass
    
    # Fallback: use centroid
    center = np.mean(all_verts, axis=0)
    shifted = all_verts - center
    dots = shifted @ normals.T
    max_dot = np.max(dots)
    s = max_dot * 2.0 / np.sqrt(3.0)
    return center[0], center[1], s


def _objective(params):
    """
    Objective function for optimization.
    params: 25 values = 12*(x, y) + angle_deg
    """
    positions = params[:24].reshape(12, 2)
    angle_deg = params[24]
    
    if not _check_separation_fast(positions):
        return 1000.0
    
    _, _, s = _compute_s_lp(positions, angle_deg)
    return s


def _hex_lattice_positions(scale=1.0):
    """Generate a hexagonal lattice arrangement of 12 unit hexagons."""
    s = scale
    h = 2.5 * s
    v = 2.165 * s  # sqrt(3)/2 * 2.5
    
    positions = np.array([
        [0.0, 0.0],
        [-h, 0.0],
        [h, 0.0],
        [-h / 2, v],
        [h / 2, v],
        [-h / 2, -v],
        [h / 2, -v],
        [-3 * h / 2, v],
        [3 * h / 2, v],
        [-3 * h / 2, -v],
        [3 * h / 2, -v],
        [0.0, -2 * v],
    ])
    return positions


def _compact_positions():
    """Generate a more compact 3-4-3-2 arrangement."""
    h = 2.5
    v = 2.165
    positions = np.array([
        [-h, 0.0], [0.0, 0.0], [h, 0.0],
        [-1.5 * h, v], [-0.5 * h, v], [0.5 * h, v], [1.5 * h, v],
        [-h, 2 * v], [0.0, 2 * v], [h, 2 * v],
        [-0.5 * h, 3 * v], [0.5 * h, 3 * v],
    ])
    return positions


def _optimized_lattice():
    """Generate an optimized lattice-like arrangement."""
    h = 2.5
    v = 2.165
    positions = np.array([
        [-1.5 * h, 0.0], [-0.5 * h, 0.0], [0.5 * h, 0.0], [1.5 * h, 0.0],
        [-2.0 * h, v], [-1.0 * h, v], [0.0, v], [1.0 * h, v],
        [-1.5 * h, 2 * v], [-0.5 * h, 2 * v], [0.5 * h, 2 * v], [1.5 * h, 2 * v],
    ])
    return positions


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon.
    Returns (inner, outer, s).
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    np.random.seed(42)
    
    best_s = float('inf')
    best_params = None
    
    # Initial configurations
    inits = []
    
    base_lattice = _hex_lattice_positions(1.0)
    inits.append(base_lattice)
    
    for scale in [0.9, 1.1]:
        inits.append(_hex_lattice_positions(scale))
    
    inits.append(_compact_positions())
    inits.append(_optimized_lattice())
    
    # Try each initial configuration with Nelder-Mead
    for init_pos in inits:
        if time.time() - start_time > deadline - 60.0:
            break
        
        init_params = np.append(init_pos.flatten(), 0.0)
        
        try:
            result = minimize(
                _objective,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_s:
                best_s = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Refine with Powell if we have a good candidate
    if best_params is not None and time.time() - start_time < deadline - 120.0:
        try:
            result = minimize(
                _objective,
                best_params,
                method='Powell',
                options={'maxiter': 2000, 'ftol': 1e-10, 'xtol': 1e-10}
            )
            if result.fun < best_s:
                best_s = result.fun
                best_params = result.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback: use the best initial configuration
        for init_pos in inits:
            if _check_separation_fast(init_pos):
                _, _, s = _compute_s_lp(init_pos, 0.0)
                if s < best_s:
                    best_s = s
                    best_params = np.append(init_pos.flatten(), 0.0)
    
    if best_params is None:
        # Last resort fallback
        init_pos = _hex_lattice_positions(1.0)
        best_params = np.append(init_pos.flatten(), 0.0)
        _, _, best_s = _compute_s_lp(init_pos, 0.0)
    
    positions = best_params[:24].reshape(12, 2)
    angle_deg = best_params[24]
    
    inner = np.zeros((12, 3))
    inner[:, 0] = positions[:, 0]
    inner[:, 1] = positions[:, 1]
    inner[:, 2] = angle_deg
    
    # Final verification
    if not _check_separation_fast(positions):
        # Try to fix by scaling up slightly
        for scale in [1.01, 1.02, 1.05]:
            positions_fixed = _hex_lattice_positions(scale)
            if _check_separation_fast(positions_fixed):
                inner[:, 0] = positions_fixed[:, 0]
                inner[:, 1] = positions_fixed[:, 1]
                inner[:, 2] = 0.0
                _, _, best_s = _compute_s_lp(positions_fixed, 0.0)
                break
    
    # Compute final outer hexagon
    outer_cx, outer_cy, s_final = _compute_s_lp(inner[:, :2], angle_deg)
    s_final += 1e-4
    
    outer = np.array([outer_cx, outer_cy, angle_deg])
    
    return inner, outer, s_final


# EVOLVE-BLOCK-END
