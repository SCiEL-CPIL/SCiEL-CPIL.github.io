import numpy as np
import time

def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _hex_edge_normals(cx, cy, s, a_deg):
    """Compute 6 outward edge normals."""
    verts = _hex_vertices(cx, cy, s, a_deg)
    normals = np.zeros((6, 2))
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        edge = v2 - v1
        n = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-15:
            n = n / norm
        mid = (v1 + v2) / 2
        center = np.array([cx, cy])
        if np.dot(n, mid - center) < 0:
            n = -n
        normals[i] = n
    return normals

def _check_separation(hex1_data, hex2_data):
    """Check if two hexagons are separated per the task definition."""
    v1 = _hex_vertices(hex1_data[0], hex1_data[1], 1.0, hex1_data[2])
    v2 = _hex_vertices(hex2_data[0], hex2_data[1], 1.0, hex2_data[2])
    
    n1 = _hex_edge_normals(hex1_data[0], hex1_data[1], 1.0, hex1_data[2])
    n2 = _hex_edge_normals(hex2_data[0], hex2_data[1], 1.0, hex2_data[2])
    
    all_normals = np.vstack([n1, n2])
    
    for n in all_normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        if max(proj1) + 1e-6 < min(proj2) or max(proj2) + 1e-6 < min(proj1):
            return True
    return False

def _compute_min_side(positions):
    """Compute minimum side of outer hexagon (centered at origin, angle 0) containing all unit hexagons."""
    all_verts = []
    for pos in positions:
        verts = _hex_vertices(pos[0], pos[1], 1.0, 0)
        all_verts.extend(verts)
    all_verts = np.array(all_verts)
    
    max_proj = 0
    for i in range(6):
        angle = np.radians(60 * i)
        normal = np.array([np.cos(angle), np.sin(angle)])
        proj = all_verts @ normal
        max_p = np.max(proj)
        if max_p > max_proj:
            max_proj = max_p
    
    return max_proj * 2.0 / np.sqrt(3)

def hexagon_packing_11():
    start_time = time.time()
    
    sqrt3 = np.sqrt(3)
    e1 = np.array([sqrt3, 0.0])
    e2 = np.array([sqrt3 / 2.0, 1.5])
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (2, -1),
    ]
    
    initial_positions = np.array([i * e1 + j * e2 for i, j in axial_coords])
    initial_params = np.concatenate([initial_positions[:, 0], initial_positions[:, 1]])
    
    best_params = initial_params.copy()
    best_side = _compute_min_side(initial_positions)
    
    # Gradient descent optimization
    def get_side(params):
        positions = np.column_stack([params[0:11], params[11:22]])
        return _compute_min_side(positions)
    
    lr = 0.05
    for iteration in range(2000):
        if time.time() - start_time > 550:
            break
        
        current_side = get_side(best_params)
        
        # Numerical gradient
        grad = np.zeros(22)
        eps = 1e-5
        for k in range(22):
            params_plus = best_params.copy()
            params_plus[k] += eps
            grad[k] = (get_side(params_plus) - current_side) / eps
        
        new_params = best_params - lr * grad
        new_side = get_side(new_params)
        
        if new_side < best_side:
            best_side = new_side
            best_params = new_params
        
        lr *= 0.995
    
    final_positions = np.column_stack([best_params[0:11], best_params[11:22]])
    final_side = _compute_min_side(final_positions)
    
    inner_data = np.column_stack([final_positions[:, 0], final_positions[:, 1], np.zeros(11)])
    outer_data = np.array([0.0, 0.0, 0.0])
    
    # Validate separation
    valid = True
    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_separation(inner_data[i], inner_data[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        inner_data = np.column_stack([initial_positions[:, 0], initial_positions[:, 1], np.zeros(11)])
        final_side = _compute_min_side(initial_positions)
    
    if final_side < np.sqrt(11):
        final_side = np.sqrt(11)
    
    return inner_data, outer_data, final_side
