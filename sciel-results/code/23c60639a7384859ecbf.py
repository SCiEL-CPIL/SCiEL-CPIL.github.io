"""Covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
SEED = 42


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _covering_radius(pts):
    """Compute the covering radius in radians for a set of points on the unit sphere.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest site. This equals the maximum circumradius of the spherical facets
    of the convex hull.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # fallback: worst case
    
    max_circ = 0.0
    for simplex in hull.simplices:
        tri = pts[simplex]
        # Compute circumcenter of the spherical triangle
        # The circumradius is the angle from the centroid direction to the farthest vertex
        # Better: compute the circumradius of the spherical triangle
        
        a, b, c = tri[0], tri[1], tri[2]
        
        # Spherical triangle circumradius
        # Use the formula: the circumcenter is the intersection of the great circle
        # perpendicular bisectors. For a triangle with vertices a, b, c on unit sphere,
        # the circumcenter u satisfies u.a = u.b = u.c = cos(R)
        
        # The circumcenter direction is proportional to (a x b) x (a x c) normalized... 
        # Actually, let's use the formula based on the plane of the triangle.
        
        # Method: find the point u on the sphere equidistant from a, b, c
        # u is in the span of a, b, c and satisfies |u-a| = |u-b| = |u-c|
        # This means u.a = u.b = u.c
        
        # The circumcenter of the spherical triangle lies along the normal to the plane
        # containing the triangle, but only if the triangle is small. More generally,
        # we solve the system.
        
        # Let u = alpha*a + beta*b + gamma*c
        # u.a = u.b = u.c
        # alpha*a.a + beta*b.a + gamma*c.a = alpha*a.b + beta*b.b + gamma*c.b
        # alpha*a.c + beta*b.c + gamma*c.c = alpha*a.b + beta*b.b + gamma*c.b
        
        # Actually, a simpler approach: the circumcenter u of a spherical triangle 
        # satisfies that u is in the span of a,b,c and u is equidistant from all three.
        
        # Let's use the cross product method. The circumcenter is along the direction:
        # (a x b) x (b x c) ... no, let me think again.
        
        # The perpendicular bisector of arc ab is the great circle through the midpoint 
        # of ab and perpendicular to it. The normal to this great circle is (a+b) normalized.
        # So u must satisfy u . (a+b) = 2*cos(d_ab/2)^2 ... this is getting complex.
        
        # Simpler: for each facet, compute the circumradius as the angle from the 
        # spherical centroid to the farthest vertex. This is an approximation.
        
        # Better exact method: The circumradius R of a spherical triangle with side 
        # lengths a, b, c (arcs) satisfies:
        # tan(R) = tan(A/2) * ... no, let me use a different approach.
        
        # The circumradius is the minimum r such that all three vertices are within 
        # angular distance r from some center. The center is the circumcenter.
        
        # Let's compute the circumcenter directly.
        # u satisfies: u.a = u.b, u.b = u.c, |u| = 1
        # u . (a - b) = 0, u . (b - c) = 0, u . u = 1
        # So u is parallel to (a-b) x (b-c)
        
        n1 = np.cross(a - b, b - c)
        if np.linalg.norm(n1) < 1e-12:
            # Degenerate
            max_circ = max(max_circ, np.arccos(np.clip(np.dot(a, b), -1, 1)))
            continue
        
        u = n1 / np.linalg.norm(n1)
        
        # The circumradius is the angle from u to a (or b or c)
        cos_R = np.dot(u, a)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        R = np.arccos(cos_R)
        
        # Check that the cap is empty (no other point inside)
        # The facet is part of the convex hull, so by definition no other point is 
        # on the other side of the plane. But we need to check the spherical cap.
        
        max_circ = max(max_circ, R)
    
    return max_circ


def _initial_fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _random_initial(seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    pts = np.array([_normalize(p) for p in pts])
    return pts


def _lloyd_relax(pts, max_iters=50, tol=1e-10):
    """Lloyd's algorithm for spherical k-means to improve the covering."""
    pts = pts.copy()
    for it in range(max_iters):
        # Compute Voronoi cells (on sphere) by assigning each random point to nearest site
        # Use a fine grid to approximate the spherical Voronoi regions
        rng = np.random.RandomState(42 + it)
        M = 5000
        test_pts = rng.randn(M, 3)
        test_pts = np.array([_normalize(p) for p in test_pts])
        
        # Assign to nearest site
        # Compute dot products
        dots = test_pts @ pts.T  # (M, N)
        assignments = np.argmax(dots, axis=1)
        
        new_pts = np.zeros_like(pts)
        for j in range(N):
            mask = assignments == j
            if np.sum(mask) > 0:
                new_pts[j] = test_pts[mask].mean(axis=0)
            else:
                new_pts[j] = pts[j]
        
        new_pts = np.array([_normalize(p) for p in new_pts])
        
        diff = np.linalg.norm(new_pts - pts)
        pts = new_pts
        
        if diff < tol:
            break
    
    return pts


def _gradient_descent(pts, max_iters=200, lr=0.01, deadline=None):
    """Gradient descent on the smooth surrogate of the covering radius.
    
    We minimize a soft-max of the angular distances from a set of test points to 
    the nearest site. This gives a smooth surrogate for the max-norm objective.
    """
    pts = pts.copy()
    rng = np.random.RandomState(42)
    
    # Use a fixed set of test points for the surrogate
    M = 2000
    test_pts = rng.randn(M, 3)
    test_pts = np.array([_normalize(p) for p in test_pts])
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute angular distances: for each test point, distance to nearest site
        # cos_dist = max_j (test_i . pts_j)
        # dist_i = arccos(cos_dist_i)
        # We want to minimize a soft-max of dist_i
        
        dots = test_pts @ pts.T  # (M, N)
        cos_dists = np.max(dots, axis=1)  # (M,)
        cos_dists = np.clip(cos_dists, -1.0 + 1e-12, 1.0 - 1e-12)
        dists = np.arccos(cos_dists)  # (M,)
        
        # Soft-max with temperature
        beta = 50.0
        # log-sum-exp for numerical stability
        max_d = np.max(dists)
        weights = np.exp(beta * (dists - max_d))
        weights /= np.sum(weights)
        
        # Gradient: for each site j, dL/dpts_j
        # L = log(sum_i exp(beta * dist_i)) approximately
        # dL/dpts_j = sum_i weights_i * beta * d(dist_i)/dpts_j
        # For test point i assigned to nearest site j_i:
        # d(dist_i)/dpts_j_i = -d_i / sin(d_i) * (test_i - cos(d_i) * pts_j_i) projected to tangent
        # Actually, dist_i = arccos(test_i . pts_j)
        # d(dist_i)/dpts_j = -test_i / sqrt(1 - (test_i . pts_j)^2) + pts_j * (test_i . pts_j) / sqrt(1 - (test_i . pts_j)^2)
        # = -(test_i - (test_i . pts_j) * pts_j) / sin(d_i)
        
        # Assign each test point to its nearest site
        assignments = np.argmax(dots, axis=1)
        
        grad = np.zeros_like(pts)
        for j in range(N):
            mask = assignments == j
            if not np.any(mask):
                continue
            w_j = weights[mask]  # weights for test points assigned to j
            i_indices = np.where(mask)[0]
            
            # For each test point i assigned to j:
            # contribution = w_i * beta * (-(test_i - cos_d * pts_j) / sin(d))
            cos_vals = cos_dists[i_indices]
            sin_vals = np.sqrt(1.0 - cos_vals ** 2 + 1e-12)
            sin_vals = np.maximum(sin_vals, 1e-8)
            
            # test_i - cos_d * pts_j
            tangent = test_pts[i_indices] - cos_vals[:, None] * pts[j]
            # Normalize tangent
            tangent_norms = np.linalg.norm(tangent, axis=1, keepdims=True)
            tangent_norms = np.maximum(tangent_norms, 1e-8)
            tangent_unit = tangent / tangent_norms
            
            # Gradient contribution
            # d(dist)/d(pts_j) = -tangent_unit / sin(d)
            coeff = w_j * beta / sin_vals
            grad[j] -= np.sum(coeff[:, None] * tangent_unit, axis=0)
        
        # Project gradient to tangent space and update
        # Ensure points stay on sphere
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-12:
            break
        
        # Move in negative gradient direction, projected to tangent
        new_pts = pts - lr * grad
        new_pts = np.array([_normalize(p) for p in new_pts])
        
        # Check for improvement
        pts = new_pts
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 20s margin before 120s limit
    
    # Start with Fibonacci spiral
    best_pts = _initial_fibonacci()
    best_score = _covering_radius(best_pts)
    
    # Try multiple random initializations and refine
    rng = np.random.RandomState(SEED)
    
    n_attempts = 15
    for attempt in range(n_attempts):
        if time.time() > deadline - 30.0:
            break
        
        # Random initialization
        init_pts = _random_initial(SEED + attempt * 1000)
        
        # Lloyd relaxation
        relaxed = _lloyd_relax(init_pts, max_iters=30, tol=1e-9)
        
        # Gradient descent refinement
        if time.time() > deadline - 20.0:
            break
        
        refined = _gradient_descent(relaxed, max_iters=100, lr=0.005, deadline=deadline - 15.0)
        
        score = _covering_radius(refined)
        if score < best_score:
            best_score = score
            best_pts = refined
    
    # Final polish: a few more gradient steps with smaller learning rate
    if time.time() < deadline - 5.0:
        final = _gradient_descent(best_pts, max_iters=50, lr=0.001, deadline=deadline)
        final_score = _covering_radius(final)
        if final_score < best_score:
            best_pts = final
    
    # Ensure all points are on the unit sphere
    best_pts = np.array([_normalize(p) for p in best_pts])
    
    return best_pts
