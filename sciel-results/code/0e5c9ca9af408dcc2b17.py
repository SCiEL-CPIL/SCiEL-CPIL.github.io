# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog


def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side d and angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _get_all_inner_vertices(positions, angles=None):
    """Return (72, 2) array of all vertices of 12 unit hexagons at given positions."""
    n = len(positions)
    if angles is None:
        angles = np.zeros(n)
    all_verts = np.empty((n * 6, 2))
    for i in range(n):
        all_verts[i*6:(i+1)*6] = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
    return all_verts


def _check_separation(positions, angles=None):
    """
    Check if all pairs of unit hexagons are separated using SAT.
    positions: (12, 2) array
    angles: (12,) array or None (all zeros)
    """
    n = len(positions)
    if angles is None:
        angles = np.zeros(n)
    
    inner_verts = [_hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]) for i in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            v1 = inner_verts[i]
            v2 = inner_verts[j]
            
            edges = []
            for k in range(6):
                edges.append(v1[(k + 1) % 6] - v1[k])
                edges.append(v2[(k + 1) % 6] - v2[k])
            
            separated = False
            for edge in edges:
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal /= norm
                
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                gap1 = min2 - max1
                gap2 = min1 - max2
                if gap1 > 1e-6 or gap2 > 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    
    return True


def _compute_s_lp(positions, angles, angle_deg):
    """
    Use linear programming to find the optimal outer center and minimum side length.
    Returns (cx, cy, s).
    """
    all_verts = _get_all_inner_vertices(positions, angles)
    a_rad = np.radians(angle_deg)
    normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    n_verts = len(all_verts)
    n_normals = 6
    
    c_matrix = np.zeros((n_verts * n_normals, 3))
    b_vec = np.zeros(n_verts * n_normals)
    
    for i in range(n_verts):
        v = all_verts[i]
        for k in range(n_normals):
            row = i * n_normals + k
            c_matrix[row, 0] = -normals[k, 0]
            c_matrix[row, 1] = -normals[k, 1]
            c_matrix[row, 2] = -1.0
            b_vec[row] = -np.dot(v, normals[k])
    
    c_obj = np.array([0.0, 0.0, 1.0])
    
    try:
        res = linprog(c_obj, A_ub=c_matrix, b_ub=b_vec, 
                     bounds=[(None, None), (None, None), (0, None)], method='highs')
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


def _build_rigid_cluster(rotation_deg, center_x, center_y):
    """
    Build a rigid cluster of 12 unit hexagons in a hexagonal close-packed arrangement.
    All hexagons have the same orientation (angle = rotation_deg).
    """
    sqrt3 = np.sqrt(3.0)
    
    # For unit hexagons with angle 0 (flat-top), the six neighbors are at:
    # The inradius is sqrt(3)/2, so center-to-center distance for edge-sharing is sqrt(3)
    # The six edge normals for flat-top hex (angle 0) are at 30, 90, 150, 210, 270, 330 degrees
    
    # Hexagonal lattice positions for 12 hexagons
    # Using a 1+6+5 pattern (center, ring 1, part of ring 2)
    local_positions = np.array([
        [0.0, 0.0],
        [sqrt3 * np.cos(np.radians(30)), sqrt3 * np.sin(np.radians(30))],
        [sqrt3 * np.cos(np.radians(90)), sqrt3 * np.sin(np.radians(90))],
        [sqrt3 * np.cos(np.radians(150)), sqrt3 * np.sin(np.radians(150))],
        [sqrt3 * np.cos(np.radians(210)), sqrt3 * np.sin(np.radians(210))],
        [sqrt3 * np.cos(np.radians(270)), sqrt3 * np.sin(np.radians(270))],
        [sqrt3 * np.cos(np.radians(330)), sqrt3 * np.sin(np.radians(330))],
        [2*sqrt3 * np.cos(np.radians(90)), 2*sqrt3 * np.sin(np.radians(90))],
        [2*sqrt3 * np.cos(np.radians(150)), 2*sqrt3 * np.sin(np.radians(150))],
        [2*sqrt3 * np.cos(np.radians(210)), 2*sqrt3 * np.sin(np.radians(210))],
        [2*sqrt3 * np.cos(np.radians(270)), 2*sqrt3 * np.sin(np.radians(270))],
        [2*sqrt3 * np.cos(np.radians(330)), 2*sqrt3 * np.sin(np.radians(330))],
    ])
    
    theta = np.radians(rotation_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    rotated_x = local_positions[:, 0] * cos_t - local_positions[:, 1] * sin_t
    rotated_y = local_positions[:, 0] * sin_t + local_positions[:, 1] * cos_t
    
    positions = np.column_stack([rotated_x + center_x, rotated_y + center_y])
    angles = np.full(12, rotation_deg)
    
    return positions, angles


def _objective_rigid(params):
    """
    Objective for rigid cluster optimization.
    params: (3,) = [rotation_deg, center_x, center_y]
    """
    rotation_deg, center_x, center_y = params
    positions, angles = _build_rigid_cluster(rotation_deg, center_x, center_y)
    
    if not _check_separation(positions, angles):
        return 1000.0
    
    _, _, s = _compute_s_lp(positions, angles, 0.0)
    return s


def _objective_free(params):
    """
    Objective for free optimization.
    params: (25,) = [x1, y1, ..., x12, y12, angle_deg]
    All inner hexagons have the same orientation (angle_deg).
    """
    positions = params[:24].reshape(12, 2)
    angle_deg = params[24]
    angles = np.full(12, angle_deg)
    
    if not _check_separation(positions, angles):
        return 1000.0
    
    _, _, s = _compute_s_lp(positions, angles, angle_deg)
    return s


def _initial_hex_lattice():
    """Generate initial hexagonal lattice positions."""
    sqrt3 = np.sqrt(3.0)
    positions = np.array([
        [0.0, 0.0],
        [sqrt3 * np.cos(np.radians(30)), sqrt3 * np.sin(np.radians(30))],
        [sqrt3 * np.cos(np.radians(90)), sqrt3 * np.sin(np.radians(90))],
        [sqrt3 * np.cos(np.radians(150)), sqrt3 * np.sin(np.radians(150))],
        [sqrt3 * np.cos(np.radians(210)), sqrt3 * np.sin(np.radians(210))],
        [sqrt3 * np.cos(np.radians(270)), sqrt3 * np.sin(np.radians(270))],
        [sqrt3 * np.cos(np.radians(330)), sqrt3 * np.sin(np.radians(330))],
        [2*sqrt3 * np.cos(np.radians(90)), 2*sqrt3 * np.sin(np.radians(90))],
        [2*sqrt3 * np.cos(np.radians(150)), 2*sqrt3 * np.sin(np.radians(150))],
        [2*sqrt3 * np.cos(np.radians(210)), 2*sqrt3 * np.sin(np.radians(210))],
        [2*sqrt3 * np.cos(np.radians(270)), 2*sqrt3 * np.sin(np.radians(270))],
        [2*sqrt3 * np.cos(np.radians(330)), 2*sqrt3 * np.sin(np.radians(330))],
    ])
    return positions


def _initial_circular(radius=2.5):
    """Generate circular initial positions."""
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return positions


def _initial_ring():
    """Generate ring initial positions."""
    sqrt3 = np.sqrt(3.0)
    positions = np.array([
        [0.0, 0.0],
        [sqrt3, 0.0],
        [-sqrt3, 0.0],
        [sqrt3/2, 1.5*sqrt3/2],
        [-sqrt3/2, 1.5*sqrt3/2],
        [sqrt3/2, -1.5*sqrt3/2],
        [-sqrt3/2, -1.5*sqrt3/2],
        [2*sqrt3, 0.0],
        [-2*sqrt3, 0.0],
        [3*sqrt3/2, 1.5*sqrt3/2],
        [-3*sqrt3/2, 1.5*sqrt3/2],
        [3*sqrt3/2, -1.5*sqrt3/2],
    ])
    return positions


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon.
    Returns (inner, outer, s).
    """
    start_time = time.time()
    deadline = start_time + 570.0
    
    best_s = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0
    
    # Strategy 1: Rigid cluster optimization (fast, 3 parameters)
    # Try multiple initial rotations
    for init_rot in [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]:
        if time.time() - start_time > deadline - 60.0:
            break
        
        init_params = np.array([init_rot, 0.0, 0.0])
        
        try:
            result = minimize(
                _objective_rigid,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_s and result.fun < 100.0:
                best_s = result.fun
                positions, angles = _build_rigid_cluster(result.x[0], result.x[1], result.x[2])
                best_positions = positions
                best_angles = angles
                best_outer_angle = 0.0
        except Exception:
            pass
    
    # Strategy 2: Free optimization (25 parameters) from multiple starts
    initial_configs = [
        _initial_hex_lattice(),
        _initial_circular(2.5),
        _initial_ring(),
    ]
    
    for config in initial_configs:
        if time.time() - start_time > deadline - 120.0:
            break
        
        # Check if initial config is valid
        angles_init = np.zeros(12)
        if not _check_separation(config, angles_init):
            continue
        
        init_params = np.append(config.flatten(), 0.0)
        
        try:
            result = minimize(
                _objective_free,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_s and result.fun < 100.0:
                best_s = result.fun
                best_positions = result.x[:24].reshape(12, 2)
                best_angles = np.full(12, result.x[24])
                best_outer_angle = result.x[24]
        except Exception:
            pass
    
    # Refinement with Powell if we have time
    if best_positions is not None and time.time() - start_time < deadline - 90.0:
        # Try to refine with Powell
        if len(best_angles) == 12 and np.allclose(best_angles, best_angles[0]):
            # Rigid-like, use 3-param refinement
            pass
        else:
            # Free refinement
            init_params = np.append(best_positions.flatten(), best_angles[0])
            try:
                result = minimize(
                    _objective_free,
                    init_params,
                    method='Powell',
                    options={'maxiter': 2000, 'ftol': 1e-10, 'xtol': 1e-10}
                )
                if result.fun < best_s and result.fun < 100.0:
                    best_s = result.fun
                    best_positions = result.x[:24].reshape(12, 2)
                    best_angles = np.full(12, result.x[24])
                    best_outer_angle = result.x[24]
            except Exception:
                pass
    
    if best_positions is None:
        # Fallback: use the hex lattice with centroid outer
        best_positions = _initial_hex_lattice()
        best_angles = np.zeros(12)
        best_outer_angle = 0.0
        _, _, best_s = _compute_s_lp(best_positions, best_angles, 0.0)
    
    # Final verification and construction
    if not _check_separation(best_positions, best_angles):
        # Fall back to a known valid configuration
        best_positions = _initial_hex_lattice()
        best_angles = np.zeros(12)
        best_outer_angle = 0.0
        _, _, best_s = _compute_s_lp(best_positions, best_angles, 0.0)
    
    cx, cy, s_final = _compute_s_lp(best_positions, best_angles, best_outer_angle)
    s_final += 1e-4  # Small buffer for numerical safety
    
    # Construct inner array: (12, 3) with (x, y, angle_deg)
    inner = np.zeros((12, 3))
    inner[:, 0] = best_positions[:, 0]
    inner[:, 1] = best_positions[:, 1]
    inner[:, 2] = best_angles
    
    # Outer: (3,) array (x, y, angle_deg)
    outer = np.array([cx, cy, best_outer_angle])
    
    return inner, outer, s_final


# EVOLVE-BLOCK-END
