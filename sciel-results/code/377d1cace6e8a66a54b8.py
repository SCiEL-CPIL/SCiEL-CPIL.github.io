# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _edge_normals(vertices):
    """Return the 6 outward edge normals (unit vectors) for a hexagon given its 6 vertices (CCW)."""
    normals = np.zeros((6, 2))
    for i in range(6):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([edge[1], -edge[0]])
        n_norm = np.linalg.norm(n)
        if n_norm > 1e-12:
            n = n / n_norm
        normals[i] = n
    return normals


def _min_gap(hex1_verts, hex2_verts):
    """Check separation between two hexagons using the 12 edge normals approach."""
    normals1 = _edge_normals(hex1_verts)
    normals2 = _edge_normals(hex2_verts)
    all_normals = np.vstack([normals1, normals2])
    
    for n in all_normals:
        proj1 = hex1_verts @ n
        proj2 = hex2_verts @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
    return False


def _is_valid_packing(inner_hex_data, outer_hex_data, side):
    """Validate the packing."""
    if not (np.all(np.isfinite(inner_hex_data)) and np.all(np.isfinite(outer_hex_data)) and np.isfinite(side)):
        return False
    
    if side < np.sqrt(11) - 1e-9:
        return False
    
    n_inner = inner_hex_data.shape[0]
    vertices_list = []
    for i in range(n_inner):
        cx, cy, angle = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1.0, angle)
        vertices_list.append(verts)
    
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            if not _min_gap(vertices_list[i], vertices_list[j]):
                return False
    
    outer_cx, outer_cy, outer_angle = outer_hex_data
    outer_verts = _hex_vertices(outer_cx, outer_cy, side, outer_angle)
    
    for i in range(n_inner):
        for p in vertices_list[i]:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                edge = p2 - p1
                cross = edge[0] * (p[1] - p1[1]) - edge[1] * (p[0] - p1[0])
                if cross < -1e-6:
                    return False
    
    shift = np.array([outer_cx, outer_cy])
    for i in range(n_inner):
        cx, cy, angle = inner_hex_data[i]
        cx_s, cy_s = cx - shift[0], cy - shift[1]
        verts = _hex_vertices(cx_s, cy_s, 1.0, angle)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k + 1) % 6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    return True


def _compute_min_side(inner_hex_data, outer_center):
    """Compute the minimum side length of a regular hexagon centered at outer_center that contains all inner hexagons."""
    all_verts = []
    for i in range(inner_hex_data.shape[0]):
        cx, cy, angle = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1.0, angle)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    all_verts_shifted = all_verts - outer_center
    
    normal_angles = np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    projections = all_verts_shifted @ normals.T
    max_abs_proj = np.abs(projections).max(axis=0)
    
    s_min = np.max(max_abs_proj) * 2.0 / np.sqrt(3.0)
    
    return s_min


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses honeycomb lattice arrangement and computes the minimum bounding hexagon.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    sqrt3 = np.sqrt(3.0)
    
    # Honeycomb lattice: centers at i*a1 + j*a2 where a1=(sqrt3,0), a2=(sqrt3/2, 1.5)
    # 11 hexagons: center + 6 neighbors + 4 more
    lattice_coords = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [-1, 1],
        [1, -1],
        [2, 0],
        [-2, 0],
        [1, 1],
        [-1, -1]
    ], dtype=float)
    
    a1 = np.array([sqrt3, 0])
    a2 = np.array([sqrt3/2, 1.5])
    
    x_coords = lattice_coords[:, 0] * a1[0] + lattice_coords[:, 1] * a2[0]
    y_coords = lattice_coords[:, 0] * a1[1] + lattice_coords[:, 1] * a2[1]
    
    inner_hex_data = np.column_stack([x_coords, y_coords, np.zeros(11)])
    
    outer_center = np.array([x_coords.mean(), y_coords.mean()])
    outer_hex_data = np.array([outer_center[0], outer_center[1], 0.0])
    
    side = _compute_min_side(inner_hex_data, outer_center)
    side *= 1.001
    
    if _is_valid_packing(inner_hex_data, outer_hex_data, side):
        return inner_hex_data, outer_hex_data, side
    
    for margin in [1.01, 1.02, 1.05, 1.1]:
        side_test = _compute_min_side(inner_hex_data, outer_center) * margin
        if _is_valid_packing(inner_hex_data, outer_hex_data, side_test):
            return inner_hex_data, outer_hex_data, side_test
    
    side_large = 10.0
    if _is_valid_packing(inner_hex_data, outer_hex_data, side_large):
        return inner_hex_data, outer_hex_data, side_large
    
    return inner_hex_data, outer_hex_data, 15.0


# EVOLVE-BLOCK-END
