# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _compute_min_side(inner_positions, outer_center, outer_angle_deg):
    """
    Compute the minimum side length s of a regular hexagon centered at outer_center
    with the given angle that contains all inner unit hexagons (with angle 0).
    """
    outer_angle = np.deg2rad(outer_angle_deg)
    normal_angles = outer_angle + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    hex0 = _hex_vertices(0, 0, 1.0, 0.0)
    all_verts = inner_positions[:, np.newaxis, :] + hex0[np.newaxis, :, :]
    all_verts = all_verts.reshape(-1, 2)
    
    rel = all_verts - outer_center
    max_projs = np.max(rel @ normals.T, axis=0)
    max_inradius = max(max_projs.max(), 0.0)
    s = 2.0 * max_inradius / np.sqrt(3.0)
    return s

def _check_separation(hex1_verts, hex2_verts, tol=1e-6):
    """Separating axis test: return True if hexagons are separated (no overlap)."""
    axes = []
    for verts in (hex1_verts, hex2_verts):
        for i in range(6):
            edge = verts[(i + 1) % 6] - verts[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                axes.append(n / norm)
    for axis in axes:
        proj1 = hex1_verts @ axis
        proj2 = hex2_verts @ axis
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return True
    return False

def _vertex_inside_outer(v, outer_verts, tol=1e-6):
    """Check if vertex v is inside or on the boundary of outer hexagon."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
        if cross < -tol:
            return False
    return True

def _validate(inner_hex_data, outer_hex_data, s):
    """Validate the packing. Return True if valid."""
    if s < np.sqrt(12) - 1e-8:
        return False
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    inner_verts_list = []
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for k in range(6):
            edge_len = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
        inner_verts_list.append(verts)
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
    for i in range(12):
        for v in inner_verts_list[i]:
            if not _vertex_inside_outer(v, outer_verts):
                return False
    return True

def _make_hex_lattice(n, offset_x=0.0, offset_y=0.0):
    """Create a honeycomb lattice arrangement of n hexagons."""
    sqrt3 = np.sqrt(3)
    # For unit hexagons in honeycomb:
    # Center distance for touching = sqrt(3)
    # Lattice: a1 = (1.5, sqrt(3)/2), a2 = (0, sqrt(3))
    positions = []
    for row in range(4):
        for col in range(4):
            x = col * 1.5 + (row % 2) * 0.75 + offset_x
            y = row * sqrt3 / 2 + offset_y
            positions.append([x, y])
    positions = np.array(positions[:n])
    center = positions.mean(axis=0)
    positions -= center
    return positions

def _make_ring(n_inner, radius, offset_x=0.0, offset_y=0.0):
    """Create hexagons arranged in a ring pattern."""
    positions = []
    if n_inner == 12:
        for i in range(6):
            a = np.deg2rad(60 * i)
            positions.append([radius * np.cos(a) + offset_x, radius * np.sin(a) + offset_y])
        for i in range(6):
            a = np.deg2rad(60 * i + 30)
            positions.append([radius * 1.8 * np.cos(a) + offset_x, radius * 1.8 * np.sin(a) + offset_y])
    positions = np.array(positions[:n_inner])
    center = positions.mean(axis=0)
    positions -= center
    return positions

def _objective(p, outer_angle_deg):
    """Objective function for optimization."""
    inner_pos = p[:24].reshape(12, 2)
    outer_center = p[24:26]
    return _compute_min_side(inner_pos, outer_center, outer_angle_deg)

def _optimize_once(init_positions, outer_angle_deg, time_budget, outer_center_init=None):
    """Run one optimization from a given initial configuration."""
    if outer_center_init is None:
        outer_center_init = np.array([0.0, 0.0])
    
    init_params = np.concatenate([init_positions.flatten(), outer_center_init])
    
    best_s = _objective(init_params, outer_angle_deg)
    best_params = init_params.copy()
    
    # Gradient descent with adaptive step size
    lr = 0.1
    for step in range(2000):
        if time.time() > time_budget:
            break
        p = best_params.copy()
        eps = 1e-5
        grad = np.zeros_like(p)
        for i in range(len(p)):
            p_plus = p.copy()
            p_plus[i] += eps
            f_plus = _objective(p_plus, outer_angle_deg)
            p_minus = p.copy()
            p_minus[i] -= eps
            f_minus = _objective(p_minus, outer_angle_deg)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        p_new = p - lr * grad
        f_new = _objective(p_new, outer_angle_deg)
        
        if f_new < best_s:
            best_s = f_new
            best_params = p_new
            lr = min(lr * 1.05, 0.5)
        else:
            lr *= 0.8
    
    # Nelder-Mead refinement
    if time.time() < time_budget:
        def obj_func(p):
            return _objective(p, outer_angle_deg)
        
        result = minimize(obj_func, best_params, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < best_s:
            best_s = result.fun
            best_params = result.x
    
    return best_params, best_s

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner_hex_data, outer_hex_data, s)
    """
    start_time = time.time()
    deadline = start_time + 570.0
    
    best_s = float('inf')
    best_params = None
    best_angle = 0.0
    
    # Generate diverse initial configurations
    inits = []
    
    # Honeycomb lattice
    inits.append(_make_hex_lattice(12))
    inits.append(_make_hex_lattice(12, offset_x=0.5, offset_y=0.3))
    inits.append(_make_hex_lattice(12, offset_x=-0.5, offset_y=-0.3))
    
    # Ring patterns
    inits.append(_make_ring(12, radius=2.0))
    inits.append(_make_ring(12, radius=2.5))
    inits.append(_make_ring(12, radius=3.0))
    
    # Random perturbations
    rng = np.random.RandomState(42)
    base = _make_hex_lattice(12)
    for _ in range(4):
        perturbed = base + rng.normal(0, 0.3, size=(12, 2))
        inits.append(perturbed)
    
    # Optimize from each initial configuration
    angles_to_try = [0, 30, 60, 90, 120, 150]
    
    for init_pos in inits:
        if time.time() > deadline - 30:
            break
        for angle_deg in angles_to_try:
            if time.time() > deadline - 10:
                break
            params, s = _optimize_once(init_pos, angle_deg, deadline - 10)
            if s < best_s:
                best_s = s
                best_params = params.copy()
                best_angle = angle_deg
    
    # Extract best solution
    if best_params is None:
        positions = _make_hex_lattice(12)
        best_params = np.concatenate([positions.flatten(), [0.0, 0.0]])
        best_s = _compute_min_side(positions, np.array([0.0, 0.0]), 0.0)
        best_angle = 0.0
    
    inner_positions = best_params[:24].reshape(12, 2)
    outer_center = best_params[24:26]
    
    # Build return arrays
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = inner_positions[i, 0]
        inner_hex_data[i, 1] = inner_positions[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([outer_center[0], outer_center[1], best_angle])
    s_final = best_s
    
    # Validate
    if not _validate(inner_hex_data, outer_hex_data, s_final):
        positions_fb = _make_hex_lattice(12)
        inner_hex_data = np.zeros((12, 3))
        for i in range(12):
            inner_hex_data[i, 0] = positions_fb[i, 0]
            inner_hex_data[i, 1] = positions_fb[i, 1]
            inner_hex_data[i, 2] = 0.0
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        s_final = _compute_min_side(positions_fb, np.array([0.0, 0.0]), 0.0) + 0.1
    
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END
