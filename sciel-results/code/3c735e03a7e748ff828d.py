# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, angle_rad):
    """Return 6 vertices of a regular hexagon centered at (cx, cy) with side length s and angle angle_rad."""
    angles = angle_rad + np.arange(6) * np.pi / 3.0
    return np.stack([cx + s * np.cos(angles), cy + s * np.sin(angles)], axis=1)


def _hex_edges(cx, cy, s, angle_rad):
    """Return edges as (p1, p2) for a regular hexagon."""
    verts = _hex_vertices(cx, cy, s, angle_rad)
    edges = []
    for i in range(6):
        edges.append((verts[i], verts[(i + 1) % 6]))
    return edges


def _check_validity(inner_hex_data, outer_hex_data, side):
    """Check if the packing is valid. Returns (is_valid, penalty)."""
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)) or not np.isfinite(side):
        return False, 1e10
    if side < np.sqrt(11) - 1e-12:
        return False, 1e10
    
    outer_x, outer_y, outer_angle_deg = outer_hex_data
    outer_angle_rad = np.radians(outer_angle_deg)
    
    # Outer hexagon vertices
    outer_verts = _hex_vertices(outer_x, outer_y, side, outer_angle_rad)
    
    # Check inner hexagons are inside outer hexagon
    # For each inner hexagon vertex, check it's inside the outer hexagon
    for i in range(11):
        cx, cy, angle_deg = inner_hex_data[i]
        angle_rad = np.radians(angle_deg)
        inner_verts = _hex_vertices(cx, cy, 1.0, angle_rad)
        
        for p in inner_verts:
            # Check if p is inside outer hexagon using cross products
            for j in range(6):
                p1 = outer_verts[j]
                p2 = outer_verts[(j + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < -1e-6:
                    return False, 1e10
    
    # Check inner hexagons don't overlap
    for i in range(11):
        for j in range(i + 1, 11):
            if _hexagons_overlap(inner_hex_data[i], inner_hex_data[j]):
                return False, 1e10
    
    # Check edge lengths are close to 1
    for i in range(11):
        cx, cy, angle_deg = inner_hex_data[i]
        angle_rad = np.radians(angle_deg)
        verts = _hex_vertices(cx, cy, 1.0, angle_rad)
        for k in range(6):
            edge_len = np.linalg.norm(verts[k] - verts[(k + 1) % 6])
            if abs(edge_len - 1.0) > 1e-8:
                return False, 1e10
    
    return True, 0.0


def _hexagons_overlap(hex1_data, hex2_data):
    """Check if two unit hexagons overlap (intersection with gap < 1e-6)."""
    cx1, cy1, a1 = hex1_data
    cx2, cy2, a2 = hex2_data
    a1_rad = np.radians(a1)
    a2_rad = np.radians(a2)
    
    # For each of the 6 edges of hex1, check if any vertex of hex2 is on the wrong side
    # And for each of the 6 edges of hex2, check if any vertex of hex1 is on the wrong side
    verts1 = _hex_vertices(cx1, cy1, 1.0, a1_rad)
    verts2 = _hex_vertices(cx2, cy2, 1.0, a2_rad)
    
    # Get edges of hex1
    for i in range(6):
        p1 = verts1[i]
        p2 = verts1[(i + 1) % 6]
        # Edge normal (pointing outward)
        edge_vec = p2 - p1
        normal = np.array([edge_vec[1], -edge_vec[0]])
        normal = normal / np.linalg.norm(normal)
        # Check if normal points away from center
        center1 = np.array([cx1, cy1])
        if np.dot(normal, p1 - center1) < 0:
            normal = -normal
        
        # Project all vertices of hex2 onto this normal
        # The hexagon is on the negative side of the edge (interior)
        # For separation, all vertices of hex2 must be on the negative side or exactly on the edge
        # Actually, we need to check if the hexagons are separated
        # The distance from edge to farthest vertex of hex2 in the normal direction
        dists = np.dot(verts2, normal) - np.dot(p1, normal)
        # The maximum distance in the positive normal direction
        max_dist = np.max(dists)
        # If max_dist > -1e-6, there might be overlap (vertex is on or past the edge)
        # But we need to be more careful: the hexagons overlap if there exists a point in both
        # A sufficient condition for separation is that for at least one edge, all vertices of the other hex
        # are strictly on the interior side.
        # Actually, let's use the separating axis theorem properly.
        # For two convex polygons, they are separated if there exists an axis along which their projections don't overlap.
        # The candidate axes are the edge normals of both polygons.
    
    # Use separating axis theorem
    axes = []
    # Edge normals of hex1
    for i in range(6):
        p1 = verts1[i]
        p2 = verts1[(i + 1) % 6]
        edge_vec = p2 - p1
        normal = np.array([-edge_vec[1], edge_vec[0]])
        normal = normal / np.linalg.norm(normal)
        axes.append(normal)
    # Edge normals of hex2
    for i in range(6):
        p1 = verts2[i]
        p2 = verts2[(i + 1) % 6]
        edge_vec = p2 - p1
        normal = np.array([-edge_vec[1], edge_vec[0]])
        normal = normal / np.linalg.norm(normal)
        axes.append(normal)
    
    for axis in axes:
        # Project all vertices of hex1
        proj1 = np.dot(verts1, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        # Project all vertices of hex2
        proj2 = np.dot(verts2, axis)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check if projections are separated (gap > 1e-6)
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return False  # They are separated along this axis
    
    # If no separating axis found, they overlap
    return True


def _objective(params, inner_base):
    """Objective function for optimization: minimize the outer hexagon side length."""
    # params: [rotation_deg, scaling, ox, oy, oangle_deg]
    # But let's fix the outer hexagon to be centered at origin with angle 0 for simplicity
    # Actually, let's optimize the rotation of the inner cluster and the outer hexagon side.
    # The outer hexagon will be centered at the centroid of the inner hexagons.
    
    rot_deg = params[0]
    scale = params[1]
    
    # Rotate and scale the inner hexagons
    theta = np.radians(rot_deg)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    inner_positions = inner_base[:, :2].copy()
    inner_angles = inner_base[:, 2].copy()
    
    # Apply rotation and scaling to positions
    inner_positions = inner_positions @ rot_mat.T * scale
    
    # The angles of the inner hexagons also rotate
    inner_angles = inner_angles + rot_deg
    
    # Center the cluster at origin
    centroid = np.mean(inner_positions, axis=0)
    inner_positions = inner_positions - centroid
    
    inner_hex_data = np.column_stack([inner_positions, inner_angles])
    
    # Find the minimum outer hexagon side length
    # We need to find the smallest regular hexagon (centered at origin, angle 0) that contains all inner hexagons.
    # For a regular hexagon with side length s centered at origin with angle 0,
    # the distance from center to each edge is s * sqrt(3) / 2.
    # The distance from center to each vertex is s.
    
    # For each inner hexagon, compute the maximum distance from origin in the direction of each of the 6 hexagon edge normals.
    # The edge normals of the outer hexagon (angle 0) are at angles 0, 60, 120, 180, 240, 300.
    
    outer_normals = np.array([
        [np.cos(np.radians(a)), np.sin(np.radians(a))] for a in range(0, 360, 60)
    ])
    
    max_required = 0.0
    
    for i in range(11):
        cx, cy, a_deg = inner_hex_data[i]
        a_rad = np.radians(a_deg)
        verts = _hex_vertices(cx, cy, 1.0, a_rad)
        
        for normal in outer_normals:
            # Project vertices onto the normal
            proj = np.dot(verts, normal)
            # The distance from origin to the farthest vertex in this direction
            max_proj = np.max(proj)
            # The outer hexagon edge in this direction is at distance s * sqrt(3) / 2 from center
            # So we need s * sqrt(3) / 2 >= max_proj
            # => s >= max_proj * 2 / sqrt(3)
            required_s = max_proj * 2.0 / np.sqrt(3.0)
            if required_s > max_required:
                max_required = required_s
    
    # Also check vertices of the outer hexagon
    # The vertices of the outer hexagon are at distance s from center, at angles 0, 60, 120, 180, 240, 300.
    # For each inner vertex, check it's inside.
    # Actually, the edge constraint should be sufficient for a convex polygon.
    
    # But we also need to ensure the inner hexagons don't overlap.
    # Check for overlaps and add penalty
    overlap_penalty = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            if _hexagons_overlap(inner_hex_data[i], inner_hex_data[j]):
                overlap_penalty += 1.0
    
    if overlap_penalty > 0:
        return max_required + overlap_penalty * 10.0
    
    return max_required


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x, y, angle_degrees)
        outer_hex_side_length: float
    """
    # Base configuration: 7 hexagons in a hexagonal cluster (center + 6 around)
    # plus 4 more.
    # Let's start with a known good configuration: the 7 hexagons in a hexagonal pattern,
    # and 4 more placed to minimize the bounding hexagon.
    
    # 7 hexagons: center at (0,0), 6 at distance 2 in directions 0, 60, 120, 180, 240, 300
    # Wait, for unit hexagons to touch, the distance between centers should be 2.
    # Let me reconsider. Two unit regular hexagons touching edge-to-edge have center distance 2.
    
    # Configuration:
    # Center: (0, 0)
    # 6 around: at angles 0, 60, 120, 180, 240, 300, distance 2
    # That's 7. Need 4 more.
    # Place them at distance 2 from two of the surrounding hexagons, extending outward.
    
    # Let's use a different approach: start with a grid and optimize.
    # Base: 11 hexagons in a roughly hexagonal shape.
    
    # Let's try: 3 rows
    # Row 1 (top): 3 hexagons
    # Row 2 (middle): 5 hexagons
    # Row 3 (bottom): 3 hexagons
    # That's 11.
    
    # For hexagonal packing, rows are offset.
    # Let's use the standard hexagonal lattice.
    # Spacing between centers in same row: 2
    # Spacing between rows: sqrt(3)
    # Offset between rows: 1
    
    # Row 0 (top, y = sqrt(3)): 3 hexagons at x = -2, 0, 2
    # Row 1 (middle, y = 0): 5 hexagons at x = -3, -1, 1, 3, 5? No, that's not centered.
    # Let's center it.
    
    # Actually, let's just use a simple configuration and let the optimizer find the best rotation.
    
    # Start with 11 hexagons in a linear-ish arrangement, then optimize.
    # Better: start with the 7 hexagonal cluster and add 4.
    
    # 7 hexagons:
    # (0, 0), (2, 0), (1, sqrt(3)), (-1, sqrt(3)), (-2, 0), (-1, -sqrt(3)), (1, -sqrt(3))
    
    # Add 4 more:
    # (3, 0), (0, 2*sqrt(3)), (-3, 0), (0, -2*sqrt(3))
    # That's 11 total.
    
    sqrt3 = np.sqrt(3)
    inner_base = np.array([
        [0, 0, 0],
        [2, 0, 0],
        [1, sqrt3, 0],
        [-1, sqrt3, 0],
        [-2, 0, 0],
        [-1, -sqrt3, 0],
        [1, -sqrt3, 0],
        [3, 0, 0],
        [0, 2*sqrt3, 0],
        [-3, 0, 0],
        [0, -2*sqrt3, 0],
    ])
    
    # Optimize rotation
    # Initial guess: rotation = 0
    x0 = np.array([0.0, 1.0])  # rotation_deg, scale (scale should be 1 for unit hexagons, but let's allow it)
    
    # Actually, the hexagons must have side length exactly 1, so scale should be 1.
    # Let's just optimize rotation.
    
    def obj_rot(params):
        rot_deg = params[0]
        return _objective(np.array([rot_deg, 1.0]), inner_base)
    
    # Try several initial rotations
    best_rot = 0
    best_obj = np.inf
    for init_rot in range(0, 60, 15):
        res = minimize(obj_rot, np.array([float(init_rot)]), method='Nelder-Mead',
                       options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
        if res.fun < best_obj:
            best_obj = res.fun
            best_rot = res.x[0]
    
    # Final rotation
    rot_deg = best_rot
    theta = np.radians(rot_deg)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    inner_positions = inner_base[:, :2] @ rot_mat.T
    inner_angles = inner_base[:, 2] + rot_deg
    
    # Center the cluster
    centroid = np.mean(inner_positions, axis=0)
    inner_positions = inner_positions - centroid
    
    inner_hex_data = np.column_stack([inner_positions, inner_angles])
    
    # Compute the outer hexagon side length
    outer_normals = np.array([
        [np.cos(np.radians(a)), np.sin(np.radians(a))] for a in range(0, 360, 60)
    ])
    
    max_required = 0.0
    for i in range(11):
        cx, cy, a_deg = inner_hex_data[i]
        a_rad = np.radians(a_deg)
        verts = _hex_vertices(cx, cy, 1.0, a_rad)
        for normal in outer_normals:
            proj = np.dot(verts, normal)
            max_proj = np.max(proj)
            required_s = max_proj * 2.0 / np.sqrt(3.0)
            if required_s > max_required:
                max_required = required_s
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = max_required
    
    # Verify validity
    is_valid, penalty = _check_validity(inner_hex_data, outer_hex_data, outer_hex_side_length)
    if not is_valid:
        # Fallback to a larger hexagon
        outer_hex_side_length = 8.0
        # Re-center
        centroid = np.mean(inner_base[:, :2], axis=0)
        inner_hex_data = inner_base - np.array([centroid[0], centroid[1], 0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
