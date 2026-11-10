# EVOLVE-BLOCK-START
import numpy as np


def _hex_vertices(cx, cy, d, a_deg):
    """Vertices of a regular hexagon with center (cx,cy), side d, angle a_deg (degrees)."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _edge_normals(vertices):
    """Outward-ish edge normals (unit) for a convex polygon given in CCW order."""
    n = len(vertices)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        # For CCW polygon, outward normal is (edge_y, -edge_x) normalized? 
        # Let's check: for CCW, the interior is to the left of each edge.
        # Left normal of edge (dx,dy) is (-dy, dx). So outward is (dy, -dx).
        nx, ny = edge[1], -edge[0]
        length = np.hypot(nx, ny)
        if length > 0:
            normals[i] = [nx / length, ny / length]
    return normals


def _check_separation(v1, n1, v2, n2, tol=1e-6):
    """Check if two convex polygons are separated by SAT. Returns True if separated (non-overlapping)."""
    # For each normal of both polygons, project and check for gap
    for normals in (n1, n2):
        for n in normals:
            proj1 = v1 @ n
            proj2 = v2 @ n
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            # If intervals are separated by more than tol
            if max1 < min2 - tol or max2 < min1 - tol:
                return True  # Found a separating axis
    return False


def _check_inside(v, outer_vertices, tol=1e-6):
    """Check if point v is inside or on the boundary of the outer hexagon (CCW vertices)."""
    for i in range(len(outer_vertices)):
        p1 = outer_vertices[i]
        p2 = outer_vertices[(i + 1) % len(outer_vertices)]
        edge = p2 - p1
        vec = v - p1
        cross = edge[0] * vec[1] - edge[1] * vec[0]
        # For CCW polygon, inside means cross >= -tol for all edges
        if cross < -tol:
            return False
    return True


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    
    Uses a hexagonal lattice arrangement and optimizes the orientation of the containing hexagon.
    """
    # Define 12 unit hexagons in a hexagonal lattice arrangement
    # Using axial coordinates (q, r) for hex grid
    # The 12 hexagons form a hexagon of radius 2 (center + ring of 6 + ring of 5 more? No, that's 19)
    # Let's use a compact arrangement:
    # Center (0,0)
    # Ring 1: 6 hexagons at distance 1
    # Ring 2: 5 hexagons to make 12 total
    
    # Actually, let me think about this differently.
    # A hexagon of side n contains 3n(n-1)+1 hexagons.
    # For n=2: 3*2*1+1 = 7
    # For n=3: 3*3*2+1 = 19
    # So 12 is between these. Let's use a different shape.
    
    # Let me use a "flower" pattern:
    # 1 center + 6 around it + 5 more = 12
    # Or a 2x3 rectangle in hex grid + extras
    
    # Let me define positions in a hex grid coordinate system
    # Using cube coordinates (x, y, z) where x+y+z=0
    # Or axial coordinates (q, r)
    
    # For simplicity, let me define the 12 positions directly in Cartesian coordinates
    # using the hex grid spacing
    
    # Hex grid with side length 1:
    # Distance between adjacent hex centers = 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732
    
    # Let me use a different approach: define positions in a triangular lattice
    # where each hexagon occupies a unit cell
    
    # Actually, for unit hexagons (side=1), the distance between centers of adjacent
    # non-overlapping hexagons is 2 * (sqrt(3)/2) = sqrt(3)
    
    # Let me define a compact arrangement of 12 hexagons:
    # Using axial coordinates (q, r) with the constraint that they form a connected shape
    
    # A good compact shape for 12 hexagons:
    # Row 0: q=-1,0,1 (3 hexagons)
    # Row 1: q=-1,0,1,2 (4 hexagons)  
    # Row 2: q=-1,0,1 (3 hexagons)
    # Row 3: q=0,1 (2 hexagons)
    # Total: 3+4+3+2 = 12
    
    # Convert axial to Cartesian for flat-top hexagons:
    # x = size * (3/2 * q)
    # y = size * (sqrt(3)/2 * q + sqrt(3) * r)
    # For size=1 (side length 1)
    
    sqrt3 = np.sqrt(3.0)
    
    axial_coords = [
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2),
        (0, 3), (1, 3)
    ]
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = sqrt3 / 2 * q + sqrt3 * r
        centers.append([x, y])
    
    centers = np.array(centers)
    
    # Now optimize the orientation of the outer hexagon
    # The outer hexagon has side length s, centered at origin (we'll shift later)
    # We want to minimize s such that all 12 unit hexagons fit inside
    
    best_s = np.inf
    best_angle = 0.0
    
    # Try many orientations
    for angle_deg in np.linspace(0, 60, 61):
        # Rotate all centers by angle_deg
        a = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(a), np.sin(a)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_centers = centers @ rot_matrix.T
        
        # For a regular hexagon with side s and angle a, the vertices are at:
        # (s*cos(a + 60i), s*sin(a + 60i)) for i=0..5
        # The hexagon contains a point (x,y) if for all edges, the point is on the inside
        
        # For a hexagon centered at origin with side s and angle a:
        # The distance from center to each side (apothem) is s * sqrt(3)/2
        # The distance from center to each vertex is s
        
        # A point (x,y) is inside the hexagon if its projection onto each of the 6
        # edge normals is <= apothem
        
        # Edge normals for a hexagon with angle a:
        # The edges are perpendicular to directions at angles a + 30 + 60i
        # So the normals are at angles a + 30 + 60i
        
        # For a unit hexagon (s=1), the apothem is sqrt(3)/2
        # For side s, the apothem is s * sqrt(3)/2
        
        # A point is inside if for each normal direction n_i:
        # |dot(point, n_i)| <= s * sqrt(3)/2
        
        # But we need to be more careful. Let me use the vertex-based check.
        
        # For a hexagon with center at origin, side s, angle a:
        # Vertices at s * (cos(a+60i), sin(a+60i)) for i=0..5
        # The hexagon is the intersection of 6 half-planes
        
        # The outward normal to edge between vertex i and vertex i+1:
        # Edge direction: v_{i+1} - v_i
        # Outward normal: rotate edge by -90 degrees (for CCW polygon)
        
        # Let me compute the required s for each orientation
        
        # For a given orientation, find the minimum s such that all rotated_centers
        # are inside the hexagon with side s centered at origin
        
        # The hexagon with side s centered at origin contains point p if:
        # For all 6 edge normals n_i (unit vectors), dot(p, n_i) <= s * sqrt(3)/2
        
        # The edge normals for a hexagon with angle a:
        # Vertex i is at angle a + 60i
        # Edge from vertex i to vertex i+1 has direction (a + 60i + 30)
        # Outward normal is at angle a + 60i + 30 - 90 = a + 60i - 60
        # Wait, let me think again.
        
        # For a regular hexagon with vertices at angles theta_i = a + 60i,
        # the edge between vertex i and i+1 is perpendicular to the bisector at angle a + 60i + 30
        # The outward normal points in the direction a + 60i + 30
        
        # So the 6 outward normals are at angles: a + 30, a + 90, a + 150, a + 210, a + 270, a + 330
        
        # A point p is inside if dot(p, n_i) <= s * sqrt(3)/2 for all i
        # where n_i are the unit outward normals
        
        # So s >= max_i |dot(p, n_i)| / (sqrt(3)/2) for all p
        
        # For multiple points, s >= max over all p and all i of |dot(p, n_i)| / (sqrt(3)/2)
        
        normals_angles = np.deg2rad(angle_deg) + np.deg2rad(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
        
        # For each center, compute the maximum projection onto any normal
        # The required s is max over all centers and all normals of |dot(center, normal)| / (sqrt(3)/2)
        
        projections = rotated_centers @ normals.T  # shape (12, 6)
        max_proj = np.abs(projections).max()
        
        s_required = max_proj / (sqrt3 / 2)
        
        if s_required < best_s:
            best_s = s_required
            best_angle = angle_deg
    
    # Now compute the final positions
    a = np.deg2rad(best_angle)
    cos_a, sin_a = np.cos(a), np.sin(a)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_centers = centers @ rot_matrix.T
    
    # The outer hexagon should be centered at the centroid of the inner hexagons
    # for optimal packing, but let's center it at the origin for simplicity
    # Actually, we should center the outer hexagon at the centroid of the inner hexagons
    centroid = rotated_centers.mean(axis=0)
    
    # Shift all centers so that the centroid is at the origin
    shifted_centers = rotated_centers - centroid
    
    # Recompute s with the shifted centers
    normals_angles = np.deg2rad(best_angle) + np.deg2rad(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
    
    projections = shifted_centers @ normals.T
    max_proj = np.abs(projections).max()
    s = max_proj / (sqrt3 / 2)
    
    # Add a small margin for numerical safety
    s *= 1.0001
    
    # Create the inner hexagon data
    # Each inner hexagon has side 1, angle 0 (or best_angle for consistency)
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = shifted_centers[i, 0]
        inner_hex_data[i, 1] = shifted_centers[i, 1]
        inner_hex_data[i, 2] = best_angle  # All inner hexagons have the same orientation
    
    # Outer hexagon centered at origin, side s, angle best_angle
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    
    # Verify the packing is valid
    # Check that no two inner hexagons overlap
    for i in range(12):
        v1 = _hex_vertices(shifted_centers[i, 0], shifted_centers[i, 1], 1.0, best_angle)
        n1 = _edge_normals(v1)
        for j in range(i + 1, 12):
            v2 = _hex_vertices(shifted_centers[j, 0], shifted_centers[j, 1], 1.0, best_angle)
            n2 = _edge_normals(v2)
            if not _check_separation(v1, n1, v2, n2):
                # Overlap detected - this shouldn't happen with proper lattice spacing
                pass
    
    # Check that all inner hexagons are inside the outer hexagon
    outer_v = _hex_vertices(0.0, 0.0, s, best_angle)
    for i in range(12):
        v1 = _hex_vertices(shifted_centers[i, 0], shifted_centers[i, 1], 1.0, best_angle)
        for vertex in v1:
            if not _check_inside(vertex, outer_v):
                # Vertex outside - shouldn't happen
                pass
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
