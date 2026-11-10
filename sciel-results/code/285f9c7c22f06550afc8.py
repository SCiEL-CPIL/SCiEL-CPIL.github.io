# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, angle a_deg."""
    a = np.radians(a_deg)
    angles = a + np.deg2rad(np.arange(6) * 60)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _check_separation(hex1, hex2, tol=1e-6):
    """
    Check if two unit hexagons are separated (no intersection).
    hex1, hex2: arrays of 6 vertices each.
    Uses the separating axis theorem: for each of the 12 edge normals,
    the projected intervals must be apart by more than tol.
    """
    for hex_a, hex_b in [(hex1, hex2), (hex2, hex1)]:
        for i in range(6):
            p1 = hex_a[i]
            p2 = hex_a[(i + 1) % 6]
            edge = p2 - p1
            # Normal to this edge
            n = np.array([-edge[1], edge[0]])
            n_len = np.linalg.norm(n)
            if n_len < 1e-12:
                continue
            n = n / n_len
            # Project both hexagons onto this normal
            proj_a = np.dot(hex_a, n)
            proj_b = np.dot(hex_b, n)
            min_a, max_a = np.min(proj_a), np.max(proj_a)
            min_b, max_b = np.min(proj_b), np.max(proj_b)
            # Check if intervals are apart
            if max_a + tol < min_b or max_b + tol < min_a:
                return True
    return False


def _check_inside_hexagon(inner_verts, outer_verts, tol=1e-6):
    """
    Check if all vertices of inner hexagons are inside or on the boundary of the outer hexagon.
    inner_verts: (11, 6, 2) array of vertices
    outer_verts: (6, 2) array of outer hexagon vertices
    """
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        # For each inner vertex, check cross product
        for vi in range(11):
            p = inner_verts[vi]
            # cross(p2 - p1, p - p1) >= -tol
            cross_vals = np.cross(edge, p - p1)
            if np.any(cross_vals < -tol):
                return False
    return True


def _is_valid(inner_data, outer_data, side, tol=1e-6):
    """Full validity check."""
    # Check finite
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if side < np.sqrt(11) - tol:
        return False
    if not np.all(np.isfinite(side)):
        return False
    
    # Get vertices
    inner_verts = np.array([
        _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        for i in range(11)
    ])
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], side, outer_data[2])
    
    # Check separation between all pairs
    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_separation(inner_verts[i], inner_verts[j], tol):
                return False
    
    # Check all inner vertices are inside outer hexagon
    if not _check_inside_hexagon(inner_verts, outer_verts, tol):
        return False
    
    # Check edge lengths are within 1e-8 of 1
    for i in range(11):
        v = inner_verts[i]
        for k in range(6):
            edge_len = np.linalg.norm(v[(k + 1) % 6] - v[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    return True


def _compute_min_side(inner_data, outer_data):
    """
    Given inner hexagon positions and outer center/angle, compute the minimum side length
    needed for the outer hexagon to contain all inner hexagons.
    """
    cx, cy = outer_data[0], outer_data[1]
    a_deg = outer_data[2]
    
    inner_verts = np.array([
        _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        for i in range(11)
    ])
    
    # Shift by outer center
    shifted_verts = inner_verts - np.array([cx, cy])
    
    # For the outer hexagon with center at origin and angle a_deg,
    # we need to find the minimum s such that all vertices are inside.
    # The outer hexagon has vertices at s*cos(a + 60i), s*sin(a + 60i)
    # The constraint is: for each edge of the outer hexagon, all inner vertices
    # must satisfy the half-plane constraint.
    
    a = np.radians(a_deg)
    angles = a + np.deg2rad(np.arange(6) * 60)
    
    min_s = np.sqrt(11)  # lower bound
    
    for i in range(6):
        p1 = s * np.array([np.cos(angles[i]), np.sin(angles[i])]) if False else None
        # Edge from vertex i to vertex i+1
        # The normal pointing inward
        v1 = np.array([np.cos(angles[i]), np.sin(angles[i])])
        v2 = np.array([np.cos(angles[(i + 1) % 6]), np.sin(angles[(i + 1) % 6])])
        edge = v2 - v1
        # Inward normal
        n = np.array([-edge[1], edge[0]])
        n_len = np.linalg.norm(n)
        if n_len < 1e-12:
            continue
        n = n / n_len
        
        # The distance from origin to the edge line is s * (1 - cos(60°)) = s * 0.5
        # Actually, the apothem of a regular hexagon with side s is s * sqrt(3)/2
        # The distance from center to each edge is s * sqrt(3)/2
        
        # For a vertex p to be inside, we need:
        # dot(p, n) <= s * sqrt(3)/2  (if n points inward)
        # Or more precisely, the constraint depends on the geometry.
        
        # Let me think more carefully. The outer hexagon centered at origin with side s
        # has vertices at distance s from center. The apothem (distance from center to edge)
        # is s * sqrt(3)/2.
        
        # For each edge, the inward normal direction n, the constraint for a point p is:
        # dot(p, n) <= apothem = s * sqrt(3)/2
        
        # So s >= 2 * dot(p, n) / sqrt(3) for all p where dot(p, n) > 0
        
        proj = np.dot(shifted_verts, n)
        max_proj = np.max(proj)
        if max_proj > 0:
            s_needed = max_proj * 2 / np.sqrt(3)
            min_s = max(min_s, s_needed)
    
    return min_s


def _local_search(inner_data, outer_data, max_iter=5000, step=0.1, deadline=None):
    """
    Local search to minimize outer hexagon side length.
    """
    best_inner = inner_data.copy()
    best_outer = outer_data.copy()
    best_side = _compute_min_side(best_inner, best_outer)
    
    if deadline is not None and time.time() > deadline:
        return best_inner, best_outer, best_side
    
    # Try random perturbations
    rng = np.random.default_rng(42)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline - 10:
            break
            
        # Perturb inner hexagons
        new_inner = best_inner.copy()
        new_outer = best_outer.copy()
        
        # Randomly perturb some inner hexagons
        n_perturb = rng.integers(1, 4)
        idx = rng.choice(11, size=n_perturb, replace=False)
        for i in idx:
            new_inner[i, 0] += rng.normal(0, step)
            new_inner[i, 1] += rng.normal(0, step)
        
        # Occasionally perturb outer angle
        if rng.random() < 0.1:
            new_outer[2] += rng.normal(0, step * 10)
        
        # Compute new side length
        new_side = _compute_min_side(new_inner, new_outer)
        
        # Check validity
        if new_side < best_side - 1e-9:
            if _is_valid(new_inner, new_outer, new_side):
                best_inner = new_inner
                best_outer = new_outer
                best_side = new_side
    
    return best_inner, best_outer, best_side


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline
    
    # Initial arrangement from seed
    inner_hex_data = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ]
    )
    
    outer_hex_data = np.array([0, 0, 0])
    
    # Compute initial side length
    side = _compute_min_side(inner_hex_data, outer_hex_data)
    
    # Verify initial validity
    if not _is_valid(inner_hex_data, outer_hex_data, side):
        # Fallback to larger hexagon
        outer_hex_data = np.array([0, 0, 0])
        side = 8.0
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
    
    best_inner = inner_hex_data.copy()
    best_outer = outer_hex_data.copy()
    best_side = side
    
    # Perform local search
    if time.time() < deadline - 30:
        best_inner, best_outer, best_side = _local_search(
            inner_hex_data, outer_hex_data, max_iter=10000, step=0.15, deadline=deadline
        )
    
    # Final verification
    if not _is_valid(best_inner, best_outer, best_side):
        # Return the known valid solution
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        best_side = 8.0
    
    return best_inner, best_outer, float(best_side)


# EVOLVE-BLOCK-END
