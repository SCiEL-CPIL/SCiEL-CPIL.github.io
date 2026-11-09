# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _hexagon_vertices(cx, cy, d, angle_deg):
    """Return the 6 vertices of a regular hexagon centered at (cx, cy) with side d and angle angle_deg."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _hexagon_edge_normals(vertices):
    """Return the outward edge normals of a convex polygon given its vertices in CCW order."""
    n = len(vertices)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        # Outward normal for CCW polygon: rotate edge by -90 degrees
        normals[i] = np.array([edge[1], -edge[0]])
        # Normalize
        norm = np.linalg.norm(normals[i])
        if norm > 1e-15:
            normals[i] /= norm
    return normals


def _separation_check(hex1_verts, hex2_verts):
    """
    Check if two convex hexagons are separated (not overlapping).
    Uses the separating axis theorem.
    Returns True if separated, False if overlapping.
    """
    # Get normals from both hexagons
    normals1 = _hexagon_edge_normals(hex1_verts)
    normals2 = _hexagon_edge_normals(hex2_verts)
    all_normals = np.vstack([normals1, normals2])
    
    for normal in all_normals:
        # Project both hexagons onto this normal
        proj1 = hex1_verts @ normal
        proj2 = hex2_verts @ normal
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check if intervals overlap
        # If max1 < min2 or max2 < min1, they are separated
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
    
    return False


def _is_inside_outer(v, outer_verts):
    """
    Check if point v is inside the outer hexagon (CCW vertices).
    For each edge (p1, p2), cross(p2-p1, v-p1) must be >= -1e-6.
    """
    n = len(outer_verts)
    for i in range(n):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % n]
        edge = p2 - p1
        vec = v - p1
        cross = edge[0] * vec[1] - edge[1] * vec[0]
        if cross < -1e-6:
            return False
    return True


def _compute_objective(params):
    """
    params = [a, b, theta, s]
    a, b: lattice basis vector lengths
    theta: angle between basis vectors (radians)
    s: outer hexagon side length
    
    Returns: negative of 1/s (to minimize) or a large penalty if invalid.
    """
    a, b, theta, s = params
    
    if s < np.sqrt(12) or s <= 0 or a <= 0 or b <= 0:
        return 1e10
    
    # Lattice basis vectors
    e1 = np.array([a, 0])
    e2 = np.array([b * np.cos(theta), b * np.sin(theta)])
    
    # Define the 12 lattice points for a hexagonal arrangement
    # Center: (0,0)
    # Ring 1: 6 points
    # Ring 2: 5 more points to make 12 total
    # Actually, let's use a standard hexagonal packing of 12 hexagons.
    # A hexagon of hexagons with radius 1 has 1 + 6 = 7.
    # With radius 2: 1 + 6 + 12 = 19. That's too many.
    # Let's use a different arrangement.
    
    # For 12 unit hexagons, a common optimal arrangement is:
    # Think of them as sitting on a triangular lattice.
    # Let's define the 12 positions in lattice coordinates (i, j):
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)  -- 6 neighbors
    # (2,0), (-2,0), (0,2), (0,-2) -- 4 more? That's 11.
    # (1,1), (-1,-1) -- 2 more? That's 13.
    
    # Let me think of a better set of 12 lattice points that form a compact cluster.
    # Actually, for hexagonal packing, the optimal arrangement of 12 hexagons
    # is often a "flower" pattern or a parallelogram.
    
    # Let's try a 2x6 parallelogram or similar.
    # Or better: use the natural hexagonal lattice and pick 12 points that
    # form a roughly hexagonal shape.
    
    # Lattice coordinates (i, j) where position = i*e1 + j*e2
    # Let's use these 12 points:
    lattice_points = [
        (0, 0),
        (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1),
    ]
    
    # Compute actual positions
    positions = []
    for i, j in lattice_points:
        pos = i * e1 + j * e2
        positions.append(pos)
    positions = np.array(positions)
    
    # All inner hexagons have angle 0, side 1
    inner_angles = np.zeros(12)
    
    # Check separation between all pairs
    for i in range(12):
        for j in range(i + 1, 12):
            verts_i = _hexagon_vertices(positions[i][0], positions[i][1], 1.0, 0)
            verts_j = _hexagon_vertices(positions[j][0], positions[j][1], 1.0, 0)
            if not _separation_check(verts_i, verts_j):
                return 1e10
    
    # Check all inner hexagons are inside the outer hexagon
    outer_verts = _hexagon_vertices(0, 0, s, 0)
    for i in range(12):
        verts_i = _hexagon_vertices(positions[i][0], positions[i][1], 1.0, 0)
        for v in verts_i:
            if not _is_inside_outer(v, outer_verts):
                return 1e10
    
    # If valid, return -1/s (minimize this)
    return -1.0 / s


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    # Initial guess
    # For a hexagonal lattice with spacing ~2 (so hexagons just touch),
    # a and b should be around 2, theta around 60 degrees.
    x0 = np.array([2.0, 2.0, np.pi / 3, 8.0])
    
    # Bounds
    bounds = [(1.5, 5.0), (1.5, 5.0), (np.pi / 4, 2 * np.pi / 3), (np.sqrt(12), 20.0)]
    
    # Optimize
    result = minimize(_compute_objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
    
    a, b, theta, s = result.x
    
    # If optimization failed, use a known valid configuration
    if not result.success or result.fun > -0.1:
        # Fallback: use the seed configuration
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
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    # Compute final positions
    e1 = np.array([a, 0])
    e2 = np.array([b * np.cos(theta), b * np.sin(theta)])
    
    lattice_points = [
        (0, 0),
        (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1),
    ]
    
    inner_hex_data = np.zeros((12, 3))
    for idx, (i, j) in enumerate(lattice_points):
        pos = i * e1 + j * e2
        inner_hex_data[idx, 0] = pos[0]
        inner_hex_data[idx, 1] = pos[1]
        inner_hex_data[idx, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(s)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
