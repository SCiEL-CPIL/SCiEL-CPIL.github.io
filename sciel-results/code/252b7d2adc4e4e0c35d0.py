"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time


def _normalize(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius in degrees.
    The covering radius is the maximum over all points on the sphere of the
    angular distance to the nearest configuration point.
    This equals the maximum circumradius of the facets of the convex hull
    of the configuration (where each facet's circumcenter lies on the sphere
    and the cap is empty).
    
    We compute this by finding, for each point, the angular distance to its
    nearest neighbors, and taking the maximum over the "Voronoi vertices"
    which are circumcenters of Delaunay triangles.
    
    A simpler and robust approach: compute the convex hull of the 3D points,
    then for each triangular face, compute the circumcenter of the three
    vertices (projected to the sphere), and check that no point is closer
    to that circumcenter than to the face's vertices. The covering radius is
    the max angular distance from such a circumcenter to its nearest point.
    
    However, a more direct approach for the covering radius:
    - For each point p_i, consider its angular distance to all other points.
    - The covering radius is determined by the largest "empty cap" center.
    
    Actually, the standard approach: the covering radius equals the maximum
    over all points y on the sphere of min_i angular_dist(y, p_i).
    The worst-case y is at a circumcenter of a Delaunay triangle (or on an
    edge if the triangle is obtuse, or at a vertex if all are obtuse).
    
    For efficiency and robustness, we'll use the following:
    1. Compute all pairwise angular distances.
    2. For each point, find its 3 nearest neighbors.
    3. For each triplet of mutually nearest neighbors, compute the circumcenter.
    4. The covering radius is the max angular distance from any such circumcenter
       to the nearest configuration point.
    
    But this is complex. A simpler and correct method:
    - The covering radius is the maximum over the circumcenters of the
      Delaunay triangulation of the spherical Delaunay.
    
    Let's use a practical approach: 
    - Compute the convex hull in 3D.
    - For each face (triangle), compute the circumcenter of the three vertices
      in 3D, normalize to get a point on the sphere.
    - The angular distance from this point to the nearest of the three vertices
      is the circumradius of that face.
    - The covering radius is the maximum such circumradius over all faces
      where the circumcenter's cap is empty (i.e., the face is a Delaunay face).
    
    For a set of points in general position, the convex hull faces that are
    part of the Delaunay triangulation will have their circumcenters as
    Voronoi vertices. The covering radius is the max circumradius.
    
    Let's implement this.
    """
    from scipy.spatial import ConvexHull
    
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # If hull fails, use a fallback
        return _covering_radius_fallback(pts)
    
    max_radius = 0.0
    
    for simplex in hull.simplices:
        # simplex has 3 indices for a triangular face in 3D
        if len(simplex) != 3:
            continue
        
        p0 = pts[simplex[0]]
        p1 = pts[simplex[1]]
        p2 = pts[simplex[2]]
        
        # Compute circumcenter of the three points in 3D
        # The circumcenter is equidistant from all three points.
        # Solve: |c - p0|^2 = |c - p1|^2 = |c - p2|^2
        # This gives a linear system.
        
        # |c|^2 - 2 c·p0 + |p0|^2 = |c|^2 - 2 c·p1 + |p1|^2
        # => 2 c·(p1 - p0) = |p1|^2 - |p0|^2
        # Similarly for p2.
        
        # Since all points are on the unit sphere, |p0| = |p1| = |p2| = 1
        # So |p1|^2 - |p0|^2 = 0
        # => c·(p1 - p0) = 0 and c·(p2 - p0) = 0
        # So c is perpendicular to (p1-p0) and (p2-p0), i.e., c is along
        # the cross product (p1-p0) x (p2-p0).
        
        v1 = p1 - p0
        v2 = p2 - p0
        cross = np.cross(v1, v2)
        
        if np.linalg.norm(cross) < 1e-12:
            continue
        
        # The circumcenter lies along the cross product direction.
        # c = t * cross for some scalar t.
        # We need |c - p0|^2 = |c|^2 - 2t c·p0 + 1 = ... 
        # Actually, |c - p0|^2 = |c|^2 - 2 c·p0 + |p0|^2
        # And we want this to be equal for all three, which it is by construction.
        # But we also need c to be the circumcenter, so |c - p0| = |c - p1| = |c - p2|.
        # This is satisfied. But we need to find the right t.
        
        # |c - p0|^2 = t^2 |cross|^2 - 2 t cross·p0 + 1
        # The circumradius R satisfies R^2 = |c - p0|^2
        # But we need c to be such that it's the circumcenter.
        # Actually, any point on the line through the origin in the direction
        # of cross is equidistant from p0, p1, p2 (since they're on a circle
        # whose plane passes through the origin... wait, not necessarily).
        
        # Let me reconsider. Three points on the unit sphere define a circle
        # (their circumcircle). The center of this circle is the circumcenter
        # of the triangle in 3D. This center is not necessarily on the sphere.
        # The "spherical" circumcenter for the covering radius is the point
        # on the sphere that is equidistant (angularly) from the three points,
        # which is the normalized circumcenter of the 3D triangle.
        
        # Let's compute the 3D circumcenter of the triangle p0, p1, p2.
        # It satisfies:
        # 2 c·(p1 - p0) = |p1|^2 - |p0|^2 = 0  (since all on unit sphere)
        # 2 c·(p2 - p0) = |p2|^2 - |p0|^2 = 0
        # So c is in the null space of the 2x3 matrix [p1-p0; p2-p0],
        # i.e., c is parallel to cross = (p1-p0) x (p2-p0).
        
        # c = t * cross
        # The distance from c to p0 is |c - p0|.
        # For c to be the circumcenter, we need |c - p0| = |c - p1| = |c - p2|.
        # This is automatically satisfied for any t since c is equidistant.
        # But the circumcenter is unique, so there's a specific t.
        
        # Actually, the circumcenter of a triangle in 3D is unique.
        # Let's solve for t properly.
        # c = t * cross
        # |c - p0|^2 = |c - p1|^2
        # This is automatically true. But we need the actual circumcenter.
        
        # The circumcenter is the point equidistant from all three vertices.
        # In the plane of the triangle, it's the intersection of perpendicular bisectors.
        # In 3D, since the three points are not collinear, the circumcenter is
        # unique and lies in the plane of the triangle.
        
        # The plane of the triangle has normal n = cross / |cross|.
        # The plane equation: n·x = n·p0 = d
        # The circumcenter c lies in this plane and satisfies c·(p1-p0) = 0 and c·(p2-p0) = 0.
        
        # Wait, I made an error. The conditions are:
        # c·(p1 - p0) = (|p1|^2 - |p0|^2)/2 = 0
        # c·(p2 - p0) = (|p2|^2 - |p0|^2)/2 = 0
        # These mean c is perpendicular to the edge vectors, so c is along the normal.
        # But the circumcenter should be in the plane of the triangle, not along the normal.
        
        # Let me re-derive. The circumcenter c satisfies:
        # |c - p0|^2 = |c - p1|^2
        # |c|^2 - 2c·p0 + |p0|^2 = |c|^2 - 2c·p1 + |p1|^2
        # -2c·p0 + 1 = -2c·p1 + 1
        # c·(p1 - p0) = 0
        
        # Similarly c·(p2 - p0) = 0.
        
        # So c is perpendicular to both (p1-p0) and (p2-p0), hence c is parallel
        # to their cross product. But this means c is along the normal to the plane
        # of the triangle. The circumcenter of a triangle in 3D is indeed along
        # the normal through the incenter of the triangle... 
        
        # Actually, I think I'm overcomplicating this. For three points on a sphere,
        # the circumcenter of the triangle (in 3D) is the center of the unique circle
        # passing through all three points. This circle lies in the plane of the triangle.
        # The center of this circle is in the plane of the triangle.
        
        # Let me use a different approach. The circumcenter c of triangle p0, p1, p2
        # can be found by solving the linear system:
        # (p1 - p0)·c = (|p1|^2 - |p0|^2)/2
        # (p2 - p0)·c = (|p2|^2 - |p0|^2)/2
        # And c lies in the plane of the triangle.
        
        # Since |p0| = |p1| = |p2| = 1, the RHS is 0.
        # So (p1 - p0)·c = 0 and (p2 - p0)·c = 0.
        # This means c is perpendicular to the plane spanned by (p1-p0) and (p2-p0),
        # i.e., c is along the normal. But this contradicts the fact that the
        # circumcenter should be in the plane.
        
        # I see the issue. The equation |c - p0|^2 = |c - p1|^2 gives:
        # -2c·p0 + |p0|^2 = -2c·p1 + |p1|^2
        # 2c·(p1 - p0) = |p1|^2 - |p0|^2 = 0
        # So c·(p1 - p0) = 0.
        
        # This is correct. It means the circumcenter is equidistant from p0 and p1,
        # so it lies on the perpendicular bisector plane of the segment p0-p1.
        # The perpendicular bisector plane of p0-p1 is the plane through the midpoint
        # (p0+p1)/2 with normal (p1-p0).
        # So c·(p1-p0) = ((p0+p1)/2)·(p1-p0) = (|p1|^2 - |p0|^2)/2 = 0.
        # So the plane passes through the origin!
        
        # This means the circumcenter is at the origin if the triangle is right-angled
        # at the origin... but the origin is not on the sphere.
        
        # I think the issue is that for points on a sphere, the circumcenter of the
        # triangle (as a 3D triangle) is not the same as the point on the sphere
        # that is equidistant (angularly) from the three points.
        
        # For the covering radius, we need the point on the sphere that maximizes
        # the minimum angular distance to the configuration. This is a Voronoi vertex
        # of the spherical Voronoi diagram, which corresponds to the circumcenter
        # of a spherical Delaunay triangle.
        
        # For a spherical Delaunay triangle with vertices p0, p1, p2, the circumcenter
        # on the sphere is the point y on the sphere such that angular_dist(y, p0) = 
        # angular_dist(y, p1) = angular_dist(y, p2).
        
        # This y is the normalized version of the 3D circumcenter of the triangle.
        # But as we saw, the 3D circumcenter of a triangle with vertices on the unit
        # sphere is along the normal to the plane of the triangle.
        
        # Let's just compute it numerically. The point y on the sphere equidistant
        # from p0, p1, p2 satisfies:
        # y·p0 = y·p1 = y·p2 = cos(theta)
        # and |y| = 1.
        
        # From y·p0 = y·p1: y·(p0 - p1) = 0
        # From y·p0 = y·p2: y·(p0 - p2) = 0
        # So y is perpendicular to (p0-p1) and (p0-p2), hence y is along
        # (p0-p1) x (p0-p2) = cross product.
        
        # So y = ± normalize((p0-p1) x (p0-p2))
        
        # We need to choose the sign such that y is in the "correct" direction
        # (the one that gives the empty cap). For the covering radius, we take
        # the one that is farthest from all points.
        
        cross = np.cross(p0 - p1, p0 - p2)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-12:
            continue
        
        y = cross / norm_cross
        
        # Check both y and -y, take the one with larger min angular distance
        def min_angular_dist(pt, all_pts):
            dots = np.dot(all_pts, pt)
            dots = np.clip(dots, -1.0, 1.0)
            return np.arccos(dots).min()
        
        d1 = min_angular_dist(y, pts)
        d2 = min_angular_dist(-y, pts)
        
        # The covering radius contribution from this face is the max of d1, d2
        # (whichever gives the larger empty cap)
        max_radius = max(max_radius, d1, d2)
    
    return np.degrees(max_radius)


def _covering_radius_fallback(pts):
    """Fallback: approximate covering radius by sampling the sphere."""
    # Sample many points on the sphere and find the max min-distance
    n_samples = 10000
    # Use Fibonacci sphere for sampling
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # For each sample, find min angular distance to any configuration point
    # dots[i, j] = samples[i] · pts[j]
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = dots.max(axis=1)  # max dot = min angle
    min_angles = np.arccos(min_dots)
    return np.degrees(min_angles.max())


def _gradient_step(pts, learning_rate, n_samples=2000):
    """
    Perform a gradient step to minimize the covering radius.
    Uses a soft-max approximation for the non-smooth max operation.
    """
    pts = _normalize(pts)
    
    # Sample points on the sphere
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # For each sample, compute the nearest configuration point
    dots = samples @ pts.T  # (n_samples, N)
    dots = np.clip(dots, -1.0, 1.0)
    
    # Angular distance = arccos(dot)
    # We want to minimize the max over samples of min over points of arccos(dot)
    # This is equivalent to minimizing the max over samples of (1 - max_j dot[sample, j])
    # approximately, for small angles.
    
    # Use soft-max for the min over points (which is max over negative dots)
    # min_j arccos(dot_j) ≈ arccos(max_j dot_j) for small angles
    # Let's use a temperature-based soft approach
    
    beta = 50.0  # temperature for soft-max
    # For each sample, the "effective" nearest point weight
    exp_neg = np.exp(-beta * dots)
    exp_neg_sum = exp_neg.sum(axis=1, keepdims=True)
    weights = exp_neg / exp_neg_sum  # (n_samples, N)
    
    # The gradient of the soft-min angular distance with respect to pts
    # This is complex, so let's use a simpler approach:
    # Move each point away from the samples for which it is the nearest
    
    # For each point p_i, compute the weighted sum of samples for which p_i is nearest
    # and move p_i away from those samples
    
    # weights[i, j] is the weight of sample i for point j
    # For point j, the "force" is the sum over samples of weights[i, j] * (samples[i] - pts[j])
    # projected onto the tangent space
    
    # Actually, to minimize the covering radius, we want to move points to cover
    # the worst-case samples. So for each point, move it toward the samples it's
    # nearest to, but in the direction that reduces the max distance.
    
    # Simpler: for each point, compute the centroid of the samples it's nearest to,
    # and move toward that centroid.
    
    # weights: (n_samples, N), weights[i, j] = how much sample i is assigned to point j
    # For point j, the weighted centroid is:
    # c_j = sum_i weights[i, j] * samples[i] / sum_i weights[i, j]
    
    weight_sums = weights.sum(axis=0)  # (N,)
    weight_sums[weight_sums < 1e-12] = 1.0
    centroids = (weights.T @ samples) / weight_sums[:, np.newaxis]  # (N, 3)
    
    # Move each point toward its centroid, projected onto the tangent space
    new_pts = np.zeros_like(pts)
    for j in range(N):
        p = pts[j]
        c = centroids[j]
        # Tangent direction: c - (c·p) p
        tangent = c - np.dot(c, p) * p
        norm_tangent = np.linalg.norm(tangent)
        if norm_tangent > 1e-12:
            tangent /= norm_tangent
        new_pts[j] = p + learning_rate * tangent
        # Renormalize
        new_pts[j] /= np.linalg.norm(new_pts[j])
    
    return new_pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    # Gradient-based refinement
    lr = 0.05
    n_samples = 3000
    
    iteration = 0
    while time.time() < deadline - 30.0:  # Leave 30s for final evaluation
        new_pts = _gradient_step(pts, lr, n_samples)
        new_radius = _covering_radius(new_pts)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            # Decrease learning rate slowly
            lr *= 0.995
        else:
            # Revert and decrease learning rate
            lr *= 0.9
            if lr < 0.001:
                break
        
        pts = new_pts
        iteration += 1
    
    # Final polish with smaller steps
    lr = 0.01
    while time.time() < deadline - 10.0:
        new_pts = _gradient_step(pts, lr, n_samples)
        new_radius = _covering_radius(new_pts)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            lr *= 0.99
        else:
            lr *= 0.8
            if lr < 0.0005:
                break
        
        pts = new_pts
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
