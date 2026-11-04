"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull

N = 11

def _hexagon_vertices(side, cx, cy, angle_rad):
    """Vertices of a regular hexagon with given side, center, and rotation."""
    # Regular hexagon has vertices at distance 'side' from center, every 60 degrees
    angles = np.linspace(0, 2*np.pi, 7, endpoint=True) + angle_rad
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack([x, y])

def _convex_hull(points):
    """Compute convex hull of 2D points. Returns hull vertices in CCW order."""
    if len(points) < 3:
        return points
    try:
        hull = ConvexHull(points)
        return points[hull.vertices]
    except Exception:
        # Fallback: use all points
        return points

def _min_enclosing_hexagon_side(hull_points, angle_rad):
    """
    Compute the minimum side length of a regular hexagon (with given rotation)
    that contains all hull_points.
    
    The containing hexagon is centered at origin for simplicity (we'll translate later).
    A regular hexagon with side s, rotation theta, centered at origin has:
    - 6 sides
    - The distance from center to each side (apothem) is s * sqrt(3) / 2
    - The sides are at angles theta + k*60 degrees for k=0..5
    
    For a point to be inside, its projection onto the normal of each side must be
    <= apothem.
    
    The minimum s is determined by the maximum over all points and all 6 side normals
    of (projection / (sqrt(3)/2)).
    """
    # Side normals: perpendicular to each side. For a hexagon with flat-top orientation
    # at rotation 0, sides are at angles 0, 60, 120, 180, 240, 300 degrees from center.
    # The outward normals point at angles 90, 150, 210, 270, 330, 30 degrees? 
    # Let me think more carefully.
    #
    # A regular hexagon with vertices at angles 0, 60, 120, ... from center (pointy-top):
    # Vertices: (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), ...
    # The side between vertex 0 and vertex 1 has a normal pointing outward.
    # Midpoint of side 0-1: (3s/4, s*sqrt(3)/4)
    # Normal direction: angle of midpoint from center = atan2(s*sqrt(3)/4, 3s/4) = atan2(sqrt(3)/4, 3/4)
    # Actually, the apothem direction (outward normal) for a regular hexagon with vertices
    # at angles 0, 60, ... is at angles 30, 90, 150, 210, 270, 330 degrees.
    # So the 6 outward normals are at angles theta + 30 + k*60 for k=0..5.
    
    normal_angles = angle_rad + np.pi/6 + np.arange(6) * np.pi/3
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # For each point, project onto each normal
    # projection = dot(point, normal)
    # We need s such that projection <= s * sqrt(3) / 2 for all points and normals
    # So s >= 2 * max(projection) / sqrt(3)
    
    # Compute all projections
    # hull_points: (M, 2), normals: (6, 2)
    projections = hull_points @ normals.T  # (M, 6)
    max_proj = np.max(projections)
    
    side = 2.0 * max_proj / np.sqrt(3.0)
    return side

def _cluster_points():
    """
    Generate the 11 unit hexagon centers in a compact cluster.
    Using a 7-hexagon flower + 4 corner hexagons from the second ring.
    
    Honeycomb lattice vectors (for hexagons of side 1, center-to-center distance is sqrt(3)):
    v1 = (sqrt(3), 0)
    v2 = (sqrt(3)/2, 3/2)
    
    7-hexagon flower centers:
    (0,0), v1, v2, v2-v1, -v1, -v2, v1-v2
    
    4 additional "corner" hexagons at:
    v1+v2, v2+(v2-v1)=2v2-v1, (v2-v1)-v1=v2-2v1, -(v1+v2)
    """
    s3 = np.sqrt(3.0)
    v1 = np.array([s3, 0.0])
    v2 = np.array([s3/2, 1.5])
    
    centers = [
        np.array([0.0, 0.0]),
        v1,
        v2,
        v2 - v1,
        -v1,
        -v2,
        v1 - v2,
        v1 + v2,
        2*v2 - v1,
        v2 - 2*v1,
        -(v1 + v2),
    ]
    
    return np.array(centers)

def _all_hexagon_vertices(centers):
    """Get all vertices of all unit hexagons in the cluster (angle 0)."""
    all_verts = []
    for c in centers:
        verts = _hexagon_vertices(1.0, c[0], c[1], 0.0)
        all_verts.append(verts)
    return np.vstack(all_verts)

# EVOLVE-BLOCK-START
def construct():
    # Generate cluster centers
    centers = _cluster_points()
    
    # Get all vertices of the 11 unit hexagons
    all_verts = _all_hexagon_vertices(centers)
    
    # Compute convex hull
    hull_points = _convex_hull(all_verts)
    
    # The cluster is not centered at origin in a way that minimizes the enclosing hexagon.
    # We need to find the optimal center and rotation.
    # Strategy: The optimal enclosing hexagon is determined by the hull.
    # For a given rotation, the minimal enclosing hexagon (of that rotation) has a center
    # that is the centroid of the "supporting" structure. But for simplicity, we can
    # note that the minimal enclosing hexagon for a given rotation will have its center
    # at the centroid of the hull, or we can optimize over center as well.
    
    # Better approach: For a given rotation angle, compute the minimal enclosing hexagon
    # of that rotation. The center should be chosen to minimize the side length.
    # For a fixed rotation, the problem reduces to: find center (cx, cy) and side s
    # such that all points are inside the hexagon, minimizing s.
    # This is equivalent to: for each of the 6 side normals, the constraint is
    #   dot(p - c, n_i) <= s * sqrt(3)/2
    #   dot(c - p, n_i) <= s * sqrt(3)/2  (for the opposite side)
    # So: 2 * |dot(p - c, n_i)| <= s * sqrt(3) for all i
    # We want to minimize s = (2/sqrt(3)) * max_i |dot(p - c, n_i)|
    
    # For a fixed rotation, this is a convex optimization in (cx, cy, s).
    # But we can simplify: the optimal center for a given rotation is such that the
    # max over the 6 directions is minimized. This is related to the Chebyshev center
    # in the L_infinity metric defined by the 6 normals.
    
    # Simpler heuristic: The optimal center is approximately the centroid of the hull.
    # Let's use the centroid of the hull points.
    
    centroid = np.mean(hull_points, axis=0)
    
    # Center the hull at origin for the rotation optimization
    hull_centered = hull_points - centroid
    
    # Now minimize over rotation angle
    def objective(theta):
        side = _min_enclosing_hexagon_side(hull_centered, theta)
        return side
    
    # Ternary search over theta in [0, pi/3) due to hexagonal symmetry
    # The function is periodic with period pi/3
    lo = 0.0
    hi = np.pi / 3.0
    iterations = 100
    for _ in range(iterations):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if objective(m1) < objective(m2):
            hi = m2
        else:
            lo = m1
    
    optimal_theta = (lo + hi) / 2.0
    optimal_side = objective(optimal_theta)
    
    # Now we need to return the positions and angles of the 11 hexagons.
    # The cluster was built with all hexagons at angle 0.
    # We rotate the entire cluster by optimal_theta.
    # The outer hexagon center is at the centroid.
    # The outer hexagon side is optimal_side, and its rotation is optimal_theta.
    
    # Rotate all centers by optimal_theta
    rot_matrix = np.array([
        [np.cos(optimal_theta), -np.sin(optimal_theta)],
        [np.sin(optimal_theta), np.cos(optimal_theta)]
    ])
    rotated_centers = centers @ rot_matrix.T
    
    # The inner hexagons each have angle 0 + optimal_theta (they rotate with the cluster)
    # But wait: the inner hexagons can be rotated independently. In our rigid cluster,
    # they all have the same rotation. That's fine.
    
    inner = []
    for c in rotated_centers:
        inner.append([c[0] + centroid[0], c[1] + centroid[1], np.degrees(optimal_theta)])
    
    outer_center = (centroid[0], centroid[1])
    outer_side = optimal_side
    outer_angle = optimal_theta
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
