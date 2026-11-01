"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere."""
    points = np.array(points)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Check for coincident points
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if np.linalg.norm(points[i] - points[j]) < 1e-10:
                return np.inf
    
    # Check if all points are in a closed hemisphere
    # This means the convex hull doesn't contain the origin
    try:
        hull = ConvexHull(points)
        # Check if origin is inside the convex hull
        # A point is inside a convex hull if it can be expressed as a convex combination
        # We check by seeing if the origin is on the same side of all facet planes
        # Or more simply: if the minimum distance from origin to any facet plane is >= 0
        # Actually, let's check if the origin is inside by checking barycentric coordinates
        # or by checking if all facet normals point away from origin
        
        # Simpler check: compute the convex hull and see if origin is inside
        # Use the fact that origin is inside iff for all facets, the origin is on the 
        # interior side. For a facet with normal n (pointing outward), origin is inside
        # if n·0 + d >= 0 where d is the offset. But scipy's ConvexHull doesn't give
        # us the offset directly in a simple way.
        
        # Alternative: check if the minimum dot product of origin with facet normals
        # Actually, let's just check if the points span R^3 and the origin is in the hull
        # by solving a linear program or checking the convex hull contains origin.
        
        # Quick check: if all points have positive dot product with some vector,
        # they're in an open hemisphere.
        # Try several random directions
        rng = np.random.RandomState(42)
        for _ in range(100):
            v = rng.randn(3)
            v = v / np.linalg.norm(v)
            if np.all(np.dot(points, v) > 0):
                return np.inf
        # Also check standard axes
        for v in [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]),
                  np.array([-1,0,0]), np.array([0,-1,0]), np.array([0,0,-1])]:
            if np.all(np.dot(points, v) > 0):
                return np.inf
        
    except Exception:
        return np.inf
    
    # Compute covering radius: max over sphere of min angular distance to nearest point
    # This is equivalent to the maximum circumradius of the facets of the convex hull
    # where each facet corresponds to a Voronoi cell on the sphere.
    
    # The covering radius is the maximum, over all points y on the sphere, of the
    # angular distance from y to its nearest configuration point.
    # This equals the maximum over all facets of the convex hull of the circumradius
    # of that facet (the radius of the smallest circle containing the facet's vertices,
    # measured as an angle from the origin).
    
    # For each facet (triangle on the convex hull), the farthest point on the sphere
    # from all three vertices is the point in the direction of the facet normal (if the
    # circumcenter is inside the triangle) or one of the vertices.
    
    # Actually, the covering radius for a set of points on the sphere is the maximum
    # angular distance from any point on the sphere to the nearest point in the set.
    # This is determined by the "empty caps" - the largest empty spherical cap.
    
    # A practical way: the covering radius is the maximum, over all facets of the
    # convex hull, of the angular radius of the smallest spherical cap centered at
    # the facet that contains no points from the set.
    
    # For a triangular facet with vertices a, b, c (unit vectors), the center of the
    # corresponding empty cap is in the direction of the normalized sum of the vertices
    # (if the triangle is acute) or in the direction of the longest edge's midpoint.
    
    # Let's compute it properly:
    # For each facet, find the point on the sphere farthest from all three vertices.
    # This is the center of the inscribed circle of the spherical triangle, or more
    # precisely, the Chebyshev center.
    
    # For three unit vectors a, b, c forming a triangle on the sphere, the point p
    # maximizing min(angle(p,a), angle(p,b), angle(p,c)) is:
    # - If the triangle is acute (all angles < 90 deg), p is in the direction of a+b+c
    #   normalized, and the covering radius for this facet is arccos((a+b+c)·p / |a+b+c|)
    #   Wait, let me think more carefully.
    
    # The angular distance from p to a is arccos(p·a). We want to maximize the minimum
    # of these. The optimal p satisfies p·a = p·b = p·c = cos(theta) for some theta.
    # This means p is equidistant from a, b, c, so p is in the direction of the
    # circumcenter of the spherical triangle.
    
    # For three points on the unit sphere, the circumcenter (in terms of direction)
    # is in the direction of a+b+c if the triangle is acute, otherwise it's in the
    # direction of the midpoint of the longest edge.
    
    # Let's just compute it numerically for each facet.
    
    facets = hull.simplices
    max_covering_radius = 0.0
    
    for facet in facets:
        a, b, c = points[facet[0]], points[facet[1]], points[facet[2]]
        
        # Find the point p on the sphere that maximizes min(angle(p,a), angle(p,b), angle(p,c))
        # This is equivalent to minimizing max(-p·a, -p·b, -p·c) subject to |p|=1
        # Or maximizing min(p·a, p·b, p·c) subject to |p|=1
        
        # The optimal p is in the direction of the normalized sum if the triangle is acute
        s = a + b + c
        norm_s = np.linalg.norm(s)
        
        if norm_s < 1e-10:
            # Degenerate case
            p = a  # fallback
        else:
            p = s / norm_s
        
        # Check if p is valid (i.e., p·a, p·b, p·c are all positive and roughly equal)
        # If not, the circumcenter is outside the triangle, and the Chebyshev center
        # is on the boundary.
        
        da = p @ a
        db = p @ b
        dc = p @ c
        
        # If all dot products are positive and similar, p is the circumcenter
        # Otherwise, we need to check the edges
        
        # The covering radius for this facet is arccos(min(da, db, dc))
        min_dot = min(da, db, dc)
        
        # But we need to make sure p is actually the optimal point. If the triangle
        # is obtuse, the Chebyshev center is the midpoint of the longest edge.
        
        # Let's check: compute the angular distances
        # If min_dot < 0, p is on the wrong side
        # Let's also check the midpoints of edges
        
        # Midpoint of edge ab
        mab = (a + b) / np.linalg.norm(a + b) if np.linalg.norm(a + b) > 1e-10 else a
        mbc = (b + c) / np.linalg.norm(b + c) if np.linalg.norm(b + c) > 1e-10 else b
        mca = (c + a) / np.linalg.norm(c + a) if np.linalg.norm(c + a) > 1e-10 else c
        
        # For each candidate center, compute the min angular distance to the three vertices
        candidates = [p, mab, mbc, mca]
        
        best_min_dot = -np.inf
        for cand in candidates:
            cand = cand / np.linalg.norm(cand)
            d_a = cand @ a
            d_b = cand @ b
            d_c = cand @ c
            # We want to maximize the minimum dot product (which corresponds to
            # minimizing the maximum angular distance)
            min_d = min(d_a, d_b, d_c)
            if min_d > best_min_dot:
                best_min_dot = min_d
        
        # The covering radius contribution from this facet is arccos(best_min_dot)
        if best_min_dot < -1:
            best_min_dot = -1
        elif best_min_dot > 1:
            best_min_dot = 1
        
        cr = np.arccos(best_min_dot) * 180.0 / np.pi
        if cr > max_covering_radius:
            max_covering_radius = cr
    
    return max_covering_radius


def _lloyd_relaxation(points, max_iters=1000, tol=1e-10):
    """Lloyd's algorithm for sphere covering: move each point to the centroid of its Voronoi cell."""
    points = np.array(points, dtype=float)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    for iteration in range(max_iters):
        # For each point, find its Voronoi region on the sphere and move to centroid
        # Approximate by: for each point, collect all other points, and move towards
        # the centroid of the empty cap.
        
        # A simpler Lloyd step: for each point i, find the nearest neighbor region
        # and move to the centroid of that region.
        
        # Compute pairwise distances
        # For each point, the Voronoi cell consists of points on the sphere closer
        # to this point than to any other.
        
        # Approximate: move each point to the centroid of the spherical Voronoi cell
        # which is approximately the normalized sum of the points in its cell.
        
        # A practical approach: for each point, compute the average of all points
        # weighted by how much they "belong" to this point's cell.
        
        # Simpler: use the fact that the optimal covering has each point at the
        # centroid of its Voronoi cell. We can approximate by moving each point
        # towards the centroid of its nearest neighbors' empty space.
        
        # Let's use a gradient-based approach instead.
        break  # Placeholder, will use gradient-based below
    
    return points


def _gradient_descent(points, max_iters=5000, lr=0.01, time_limit=100):
    """Gradient descent to minimize the covering radius."""
    points = np.array(points, dtype=float)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute gradient of covering radius
        # The covering radius is determined by the facet with the largest circumradius.
        # We can compute the gradient with respect to each point.
        
        # For now, use a simple subgradient: for each point, compute the direction
        # that would reduce the covering radius.
        
        # A simpler approach: for each point, compute the force from the nearest
        # "empty" direction.
        
        # Let's use a different approach: minimize the sum of squared covering radii
        # over a grid of test points.
        
        # Generate test points on the sphere
        n_test = 2000
        rng = np.random.RandomState(42)
        test_points = rng.randn(n_test, 3)
        test_points = test_points / np.linalg.norm(test_points, axis=1, keepdims=True)
        
        # Compute distances from test points to all configuration points
        # dist[i, j] = angular distance from test_points[i] to points[j]
        dots = np.einsum('ij,kj->ik', test_points, points)
        dots = np.clip(dots, -1, 1)
        angles = np.arccos(dots)  # radians
        
        # For each test point, the distance to nearest configuration point
        nearest_dist = np.min(angles, axis=1)
        
        # The covering radius is approximated by the max of nearest_dist
        # We want to minimize this. Use a smooth approximation: log-sum-exp
        beta = 50.0
        log_sum_exp = np.log(np.sum(np.exp(beta * nearest_dist))) / beta
        loss = log_sum_exp
        
        # Compute gradient with respect to each configuration point
        # d(loss)/d(points[j]) = sum_i [ (beta * exp(beta * nearest_dist[i]) / sum_k exp(beta * nearest_dist[k])) * d(nearest_dist[i])/d(points[j]) ]
        # where d(nearest_dist[i])/d(points[j]) is non-zero only if points[j] is the nearest to test_points[i]
        
        nearest_idx = np.argmin(angles, axis=1)
        
        # Gradient
        grad = np.zeros_like(points)
        weights = np.exp(beta * nearest_dist)
        weights /= np.sum(weights)
        
        for i in range(n_test):
            j = nearest_idx[i]
            # d(angular_distance)/d(point) = -test_point / (|test_point - point| * sin(angle))
            # = -test_point / (sqrt(2 - 2*cos(angle)) * sin(angle))
            # = -test_point / (2 * sin(angle/2) * sin(angle))
            # = -test_point / (2 * sin(angle/2) * 2 * sin(angle/2) * cos(angle/2))
            # = -test_point / (4 * sin^2(angle/2) * cos(angle/2))
            
            angle = nearest_dist[i]
            if angle < 1e-6:
                continue
            sin_half = np.sin(angle / 2)
            cos_half = np.cos(angle / 2)
            if sin_half < 1e-8 or cos_half < 1e-8:
                continue
            factor = weights[i] * beta / (4 * sin_half**2 * cos_half)
            grad[j] -= factor * test_points[i]
        
        # Project gradient to tangent space and update
        for j in range(N):
            # Project grad[j] to tangent space at points[j]
            tangent = grad[j] - np.dot(grad[j], points[j]) * points[j]
            points[j] += lr * tangent
            # Renormalize
            points[j] = points[j] / np.linalg.norm(points[j])
        
        # Evaluate
        cr = _covering_radius(points)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    return best_points


def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    start_time = time.time()
    deadline = start_time + 140  # 140 second deadline (10s safety margin)
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    # Try multiple random restarts with gradient descent
    rng = np.random.RandomState(123)
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            pts = points.copy()
        else:
            # Random perturbation
            pts = points + 0.1 * rng.randn(N, 3)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Gradient descent
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        
        time_limit = max(1, time_remaining - 10)
        pts = _gradient_descent(pts, max_iters=3000, lr=0.005, time_limit=time_limit)
        
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_points = pts.copy()
    
    # Final polish with smaller learning rate
    if time.time() < deadline - 10:
        time_remaining = deadline - time.time() - 5
        if time_remaining > 5:
            pts = _gradient_descent(best_points, max_iters=2000, lr=0.001, time_limit=time_remaining)
            cr = _covering_radius(pts)
            if cr < best_cr:
                best_cr = cr
                best_points = pts.copy()
    
    return best_points
# EVOLVE-BLOCK-END
