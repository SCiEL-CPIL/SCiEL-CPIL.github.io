"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize_scalar
import math

N = 12

# EVOLVE-BLOCK-START
def _hex_corners(cx, cy, side, angle_deg):
    """Return 6 corners of a regular hexagon with center (cx,cy), side length, rotation."""
    angle = math.radians(angle_deg)
    corners = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        corners.append([x, y])
    return np.array(corners)

def _min_enclosing_hex_side(points, outer_angle_deg):
    """
    Given a set of points (representing vertices of all inner hexagons),
    find the minimum side length of a regular hexagon with given rotation
    that contains all points, centered optimally.
    
    For a regular hexagon with side s and rotation phi, the constraint for a point p
    to be inside is: for each of 6 directions (normal to each edge), the projection
    of (p - center) onto that normal must be <= s * sqrt(3)/2 * something...
    
    Actually, for a regular hexagon with side length s, the distance from center to
    each edge (apothem) is s * sqrt(3)/2.
    
    So for each of the 6 edge normals n_i (unit vectors), we need:
    (p - c) · n_i <= s * sqrt(3)/2 for all p and all i
    and (p - c) · (-n_i) <= s * sqrt(3)/2
    
    This gives: |(p - c) · n_i| <= s * sqrt(3)/2
    
    So s >= max over all p, i of |(p - c) · n_i| / (sqrt(3)/2)
    
    To minimize s over c, we need to choose c to minimize the maximum projection.
    This is equivalent to finding the minimum enclosing hexagon with fixed orientation.
    """
    phi = math.radians(outer_angle_deg)
    # 6 edge normal directions for the outer hexagon
    normals = []
    for i in range(6):
        theta = phi + i * math.pi / 3
        normals.append(np.array([math.cos(theta), math.sin(theta)]))
    normals = np.array(normals)
    
    # Project all points onto each normal
    # projections shape: (len(points), 6)
    projections = points @ normals.T
    
    # For each normal direction i, we need:
    # min_proj_i <= (c · n_i) <= max_proj_i
    # The width in direction i is (max_proj_i - min_proj_i)
    # The constraint is that there exists c such that for all i:
    #   c · n_i is in [min_proj_i + s*sqrt(3)/2, max_proj_i - s*sqrt(3)/2]
    # Wait, let me reconsider.
    
    # For point p to be inside hexagon centered at c with side s and orientation phi:
    # For each normal n_i: (p - c) · n_i <= s * sqrt(3)/2
    # And (c - p) · n_i <= s * sqrt(3)/2, i.e., (p - c) · n_i >= -s * sqrt(3)/2
    # So |(p - c) · n_i| <= s * sqrt(3)/2
    # This means: c · n_i must be in [p · n_i - s*sqrt(3)/2, p · n_i + s*sqrt(3)/2] for all p
    # So c · n_i must be in [max_p(p · n_i) - s*sqrt(3)/2, min_p(p · n_i) + s*sqrt(3)/2]
    # Wait no: c · n_i >= p · n_i - s*sqrt(3)/2 for all p => c · n_i >= max_p(p · n_i) - s*sqrt(3)/2
    # And c · n_i <= p · n_i + s*sqrt(3)/2 for all p => c · n_i <= min_p(p · n_i) + s*sqrt(3)/2
    # So we need: max_p(p · n_i) - s*sqrt(3)/2 <= c · n_i <= min_p(p · n_i) + s*sqrt(3)/2
    # This requires: max_p(p · n_i) - s*sqrt(3)/2 <= min_p(p · n_i) + s*sqrt(3)/2
    # => max_p(p · n_i) - min_p(p · n_i) <= s * sqrt(3)
    # => s >= (max_p(p · n_i) - min_p(p · n_i)) / sqrt(3)
    
    # So for a fixed orientation, the minimum s is:
    # s = max_i [ (max_p(p · n_i) - min_p(p · n_i)) / sqrt(3) ]
    
    widths = projections.max(axis=0) - projections.min(axis=0)
    s = widths.max() / math.sqrt(3)
    
    # Now find center c. We need c such that for all i:
    # max_p(p · n_i) - s*sqrt(3)/2 <= c · n_i <= min_p(p · n_i) + s*sqrt(3)/2
    # Let's pick c · n_i = (max_p(p · n_i) + min_p(p · n_i)) / 2 for each i
    # This may not be consistent, but we can solve for c.
    
    # Actually, for the return value we just need the side length and a valid center.
    # Let's compute center as the average of the midpoints in each normal direction.
    midpoints_proj = (projections.max(axis=0) + projections.min(axis=0)) / 2
    # c = normals^T * midpoints_proj (least squares or direct if normals span R2)
    # normals is 6x2, so we solve normals @ c = midpoints_proj in least squares
    c, _, _, _ = np.linalg.lstsq(normals, midpoints_proj, rcond=None)
    
    return s, c

def _cluster_points(cluster_coords, rotation_deg, scale=1.0):
    """
    Given hex grid coordinates (axial or similar), convert to Cartesian,
    apply rotation and scale, return all vertex points of the hexagons.
    """
    # Convert hex axial coords to Cartesian for pointy-top hexagons
    # Axial (q, r) to Cartesian:
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    points = []
    for q, r in cluster_coords:
        cx = math.sqrt(3) * (q + r / 2.0) * scale
        cy = 1.5 * r * scale
        # Rotate
        theta = math.radians(rotation_deg)
        rx = cx * math.cos(theta) - cy * math.sin(theta)
        ry = cx * math.sin(theta) + cy * math.cos(theta)
        # Get all 6 vertices of this unit hexagon (pointy-top, side=1)
        corners = _hex_corners(rx, ry, 1.0, rotation_deg)
        points.append(corners)
    
    return np.vstack(points)

def _try_cluster(cluster_coords):
    """Try a cluster shape, optimize rotation, return best (side, inner, center, outer_angle)."""
    def objective(theta_deg):
        pts = _cluster_points(cluster_coords, theta_deg)
        # Also try outer hexagon at various angles, but let's first fix outer angle to 0
        # and just rotate the cluster. Then we can also optimize outer angle.
        s, c = _min_enclosing_hex_side(pts, 0.0)
        return s
    
    # Optimize rotation of cluster
    res = minimize_scalar(objective, bounds=(0, 180), method='bounded', options={'xatol': 1e-6})
    best_theta = res.x
    best_s = res.fun
    
    # Now also optimize outer hexagon angle
    def objective2(phi_deg):
        pts = _cluster_points(cluster_coords, best_theta)
        s, c = _min_enclosing_hex_side(pts, phi_deg)
        return s
    
    res2 = minimize_scalar(objective2, bounds=(0, 60), method='bounded', options={'xatol': 1e-6})
    best_outer_angle = res2.x
    best_s2 = res2.fun
    
    if best_s2 < best_s:
        best_s = best_s2
    
    # Reconstruct
    pts = _cluster_points(cluster_coords, best_theta)
    s, c = _min_enclosing_hex_side(pts, best_outer_angle)
    
    # Build inner list
    inner = []
    for q, r in cluster_coords:
        cx = math.sqrt(3) * (q + r / 2.0)
        cy = 1.5 * r
        theta = math.radians(best_theta)
        rx = cx * math.cos(theta) - cy * math.sin(theta)
        ry = cx * math.sin(theta) + cy * math.cos(theta)
        inner.append([rx, ry, best_theta])
    
    return best_s, inner, c, best_outer_angle

# Define several cluster shapes (axial coordinates for pointy-top hexagons)
# 3x4 parallelogram
cluster_3x4 = [(q, r) for q in range(3) for r in range(4)]

# 2x6 parallelogram  
cluster_2x6 = [(q, r) for q in range(2) for r in range(6)]

# 4x3 parallelogram (different orientation)
cluster_4x3 = [(q, r) for q in range(4) for r in range(3)]

# Try a more compact shape: hexagonal-ish
# Center + 6 around + 5 more? Let's try a few
# Shape: rows of 3,3,3,3
cluster_rows3 = [(q, r) for r in range(4) for q in range(3)]

# Let's also try offset shapes
# A "brick wall" pattern
cluster_brick = [(q, r) for r in range(4) for q in range(3)]  # same as above for now

clusters = [cluster_3x4, cluster_2x6, cluster_4x3, cluster_rows3]

def construct():
    best_result = None
    best_score = float('inf')
    
    for cluster in clusters:
        try:
            s, inner, center, outer_angle = _try_cluster(cluster)
            if s < best_score:
                best_score = s
                best_result = (inner, tuple(center), s, outer_angle)
        except Exception:
            continue
    
    if best_result is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END

