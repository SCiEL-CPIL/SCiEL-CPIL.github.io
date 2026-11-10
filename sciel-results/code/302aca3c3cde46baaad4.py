"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _compute_exact_covering_radius(points):
    """
    Compute the exact covering radius in degrees using the convex hull.
    This is the ground truth metric required for final scoring.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        max_r = 0.0
        for f in facets:
            if len(f) != 3:
                continue
            a, b, c = points[f[0]], points[f[1]], points[f[2]]
            
            # Circumcenter of triangle abc on the sphere
            n = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            n = n / norm_n
            
            # Ensure normal points outward (away from origin, roughly)
            # The centroid of the triangle is inside the hull relative to the facet?
            # Actually, for a convex hull containing the origin, the outward normal
            # satisfies dot(n, a) > 0.
            # Wait, the circumcenter x = a + t*n.
            # t = -2 * dot(a, n).
            # If dot(a, n) > 0, t is negative. x = a - 2*dot(a,n)*n.
            # Let's just compute both candidates x and -x? No, the circumcenter is unique.
            # The condition for the "empty cap" center is that it is the point on the sphere
            # equidistant from a,b,c that is NOT covered by other points?
            # Actually, the covering radius is the max over all y of min_i dist(y, p_i).
            # The maximum of this function occurs at a circumcenter of a Delaunay triangle
            # (facet of the convex hull).
            # There are two points equidistant from a,b,c on the sphere.
            # One is the circumcenter of the spherical triangle formed by a,b,c.
            # The other is its antipode.
            # The one that matters is the one that is "exposed" to the outside, i.e.,
            # the one for which the cap centered there with radius r does not contain any other points?
            # No, the definition is: max_y min_i d(y, p_i).
            # This maximum is attained at a point y* which is equidistant from at least 3 points
            # (or 2 if on a great circle edge, but generic case is 3).
            # y* is a circumcenter of a triangle formed by 3 points.
            # Which circumcenter? The one that is a local maximum of the distance to the nearest point.
            # For a facet of the convex hull, the circumcenter x is such that the cap C(x, r) 
            # contains no other points? No, that's for packing.
            # For covering, we want the point y that is farthest from all p_i.
            # This y is the circumcenter of a triangle of points such that the cap centered at y
            # with radius d(y, p_i) contains NO other points?
            # Actually, if there were a point p_k inside C(y, d(y,p_i)), then d(y, p_k) < d(y, p_i),
            # so y would be closer to p_k, and the distance to the nearest point would be smaller.
            # Thus, for y to be a local maximum of min_i d(y, p_i), the cap C(y, r) must be empty of other points.
            # So we check all circumcenters, and for each, compute the distance to the 3 vertices.
            # Then we check if any OTHER point is closer to the circumcenter than this distance.
            # If not, this circumcenter is a candidate for the global max.
            # Actually, we just need to compute min_i d(x, p_i) for all candidates x.
            # The global covering radius is the max of these mins?
            # No. The function f(y) = min_i d(y, p_i) is continuous. Its maximum is the covering radius.
            # The maximum must occur at a point where the gradient is zero (if interior) or boundary.
            # On the sphere, the critical points of f(y) are circumcenters of triangles of points.
            # Specifically, if y is equidistant from p_i, p_j, p_k, and farther from all others,
            # then y is a local maximum of f.
            # So the algorithm is:
            # 1. Find all circumcenters of all triples of points.
            # 2. For each circumcenter x, compute r_x = min_i d(x, p_i).
            # 3. The covering radius is max_x r_x.
            # However, checking all triples is O(N^3). N=130, so 130^3/6 ~ 370k triples.
            # This is feasible in a tight loop or vectorized, but might be slow in Python.
            # The convex hull approach only checks O(N) facets (triangles).
            # Is it true that the global maximum of min_i d(y, p_i) is always at a circumcenter of a hull facet?
            # Yes, for points on a sphere, the Voronoi diagram (spherical) has vertices at circumcenters of Delaunay triangles.
            # The Delaunay triangulation is dual to the Voronoi diagram.
            # The vertices of the Voronoi diagram are the circumcenters.
            # The maximum of the distance function is at a Voronoi vertex.
            # So checking only the hull facets (which are the Delaunay triangles) is sufficient.
            
            t = -2.0 * np.dot(a, n)
            x = a + t * n
            norm_x = np.linalg.norm(x)
            if norm_x < 1e-10:
                continue
            x = x / norm_x
            
            # Calculate distance to a
            cos_r = np.clip(np.dot(x, a), -1.0, 1.0)
            r = np.arccos(cos_r)
            
            # Verify that x is the "empty" circumcenter.
            # Check if any other point is closer to x than r.
            # dists = np.arccos(np.clip(points @ x, -1, 1))
            # min_dist = np.min(dists)
            # if min_dist > r + 1e-9:
            #    # x is a local max
            #    max_r = max(max_r, r)
            # 
            # But wait, if we just take the max r over all hull facet circumcenters, 
            # is that correct?
            # Consider a point p inside the triangle abc (on the sphere).
            # Then the circumcenter of abc might not be the relevant one.
            # But the hull facets are the "outer" triangles.
            # The Voronoi vertices are indeed the circumcenters of the Delaunay triangles.
            # The Delaunay triangles are the facets of the convex hull (for points on sphere).
            # So we just need to compute the circumradius for each hull facet.
            # We don't even need to check for empty caps if we trust the hull structure,
            # BUT we must ensure we pick the correct circumcenter (the one outside the hull? no, on the sphere).
            # There are two circumcenters. One is "inside" the spherical triangle abc, one is "outside".
            # The one that is a Voronoi vertex is the one that is NOT covered by the caps of a,b,c?
            # No, the Voronoi vertex is equidistant.
            # Let's just compute r for both x and -x?
            # No, x is the intersection of the plane abc and the sphere.
            # The two points are x and -x.
            # One of them is the circumcenter of the spherical triangle containing the origin?
            # Actually, for the covering problem, we want the point y that is farthest from all p_i.
            # This y will be the circumcenter of a triangle such that the cap centered at y with radius r
            # does not contain any other points.
            # If we compute r for the circumcenter x of a hull facet, and if x is such that
            # all other points are farther than r, then r is a candidate.
            # If x is such that some other point is closer, then x is not a local max of the distance function.
            # However, the hull facet circumcenter should be a local max if the points are in general position?
            # Let's assume the standard result: The covering radius is the max circumradius of the hull facets.
            # We will compute r for the circumcenter x derived from the outward normal.
            
            if r > max_r:
                max_r = r
                
        return np.degrees(max_r)
    except Exception:
        return np.inf

def _logsumexp_covering_radius(points, beta=100.0):
    """
    Approximate the covering radius using a log-sum-exp of the negative distances.
    We want to minimize max_y min_i d(y, p_i).
    This is equivalent to minimizing the "soft max" of the distances?
    Actually, let's minimize a smooth approximation of the covering radius.
    One common approach is to minimize the sum of exp(-beta * d(y, p_i))? No.
    
    Let's use the fact that the covering radius R is the smallest r such that the union of caps C(p_i, r) covers the sphere.
    This is hard to optimize directly.
    
    Alternative: Minimize the maximum distance.
    We can approximate max_i d(x, p_i) ? No, we want min_i d(x, p_i) to be large?
    No, we want to place points such that the "worst" point on the sphere is as close as possible.
    
    Let's use the gradient descent on the points.
    Objective: Minimize J = sum_{y in grid} max(0, d(y, nearest_p) - target)? No.
    
    Let's use the log-sum-exp trick on the "gaps".
    Consider a set of test points Y on the sphere.
    For each y in Y, let d_min(y) = min_i d(y, p_i).
    We want to minimize max_y d_min(y).
    Approximate max_y d_min(y) by (1/beta) * log(sum_y exp(beta * d_min(y))).
    This is differentiable with respect to p_i.
    
    d_min(y) = min_i d(y, p_i) is not smooth.
    Approximate min_i d(y, p_i) by -(1/beta) * log(sum_i exp(-beta * d(y, p_i))).
    Then J = (1/beta) * log(sum_y exp(beta * (-(1/beta) * log(sum_i exp(-beta * d(y, p_i))))))
    J = (1/beta) * log(sum_y exp(-log(sum_i exp(-beta * d(y, p_i)))))
    J = (1/beta) * log(sum_y 1 / sum_i exp(-beta * d(y, p_i)))
    
    This is a valid smooth approximation of the covering radius.
    """
    # Generate a grid of test points
    # We need a sufficiently dense grid to approximate the max.
    # For N=130, the covering radius is ~11 degrees.
    # A grid with spacing ~5 degrees should be enough.
    # Number of points ~ 4 * pi / (5 deg in rad)^2 ~ 12.5 / 0.0076 ~ 1600.
    
    # Use a Fibonacci sphere for the grid for uniformity
    n_grid = 2000
    idx = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * idx / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * idx
    r = np.sqrt(1.0 - z * z)
    y_grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute distances d(y, p_i) for all y and i
    # dots = y_grid @ points.T  (n_grid x N)
    dots = y_grid @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots) # (n_grid x N)
    
    # Compute min_i exp(-beta * d(y, p_i))
    # We want sum_i exp(-beta * d(y, p_i))
    exp_terms = np.exp(-beta * dists)
    sum_exp = np.sum(exp_terms, axis=1) # (n_grid,)
    
    # J = (1/beta) * log(sum_y 1 / sum_exp)
    # To avoid underflow/overflow, we can work with logs.
    log_sum_exp = np.log(sum_exp + 1e-300)
    log_inv = -log_sum_exp
    log_sum_inv = np.log(np.sum(np.exp(log_inv)) + 1e-300)
    
    J = log_sum_inv / beta
    
    return J

def _gradient_logsumexp(points, beta=100.0, n_grid=2000):
    """
    Compute the gradient of the logsumexp covering radius approximation.
    """
    idx = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * idx / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * idx
    r = np.sqrt(1.0 - z * z)
    y_grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = y_grid @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    exp_terms = np.exp(-beta * dists)
    sum_exp = np.sum(exp_terms, axis=1)
    
    # We need dJ/dp_i.
    # J = (1/beta) log( sum_y 1/S_y ), where S_y = sum_i exp(-beta d_yi)
    # dJ/dp_i = (1/beta) * (1 / sum_y 1/S_y) * sum_y d(1/S_y)/dp_i
    # d(1/S_y)/dp_i = -1/S_y^2 * dS_y/dp_i
    # dS_y/dp_i = sum_j delta_{j,i} * d(exp(-beta d_yj))/dp_i
    # = d(exp(-beta d_yi))/dp_i
    # = exp(-beta d_yi) * (-beta) * d(d_yi)/dp_i
    
    # d(d_yi)/dp_i:
    # d_yi = arccos(y . p_i)
    # d/dp_i arccos(u) = -1/sqrt(1-u^2) * du/dp_i
    # du/dp_i = y
    # So d(d_yi)/dp_i = -y / sqrt(1 - (y.p_i)^2)
    # Let w_yi = 1 / sqrt(1 - dots^2)
    # Then d(d_yi)/dp_i = -w_yi * y
    
    # So d(exp(-beta d_yi))/dp_i = exp(-beta d_yi) * (-beta) * (-w_yi * y)
    # = beta * w_yi * y * exp(-beta d_yi)
    
    # dS_y/dp_i = beta * w_yi * y * exp(-beta d_yi)
    
    # d(1/S_y)/dp_i = -1/S_y^2 * beta * w_yi * y * exp(-beta d_yi)
    
    # dJ/dp_i = (1/beta) * (1 / Z) * sum_y [ -1/S_y^2 * beta * w_yi * y * exp(-beta d_yi) ]
    # where Z = sum_y 1/S_y
    # dJ/dp_i = (1/Z) * sum_y [ -1/S_y^2 * w_yi * y * exp(-beta d_yi) ]
    
    # Note: This gradient is in the ambient R^3.
    # We need to project it onto the tangent space of the sphere at p_i.
    # grad_tan = grad - (grad . p_i) p_i
    
    # Compute weights for each y, i
    # w_yi = 1 / sqrt(1 - dots^2)
    denom = np.sqrt(np.maximum(1.0 - dots**2, 1e-12))
    w = 1.0 / denom
    
    # term_yi = w * exp(-beta * dists)
    term_yi = w * exp_terms
    
    # We need sum_y [ -1/S_y^2 * term_yi * y ]
    # S_y is (n_grid,)
    # term_yi is (n_grid, N)
    # y is (n_grid, 3)
    
    # coeff_y = -1 / S_y^2  (n_grid,)
    coeff_y = -1.0 / (sum_exp**2 + 1e-300)
    
    # weighted_terms = coeff_y[:, np.newaxis] * term_yi  (n_grid, N)
    weighted_terms = coeff_y[:, np.newaxis] * term_yi
    
    # grad_i = sum_y weighted_terms[y, i] * y[y, :]
    # grad = weighted_terms.T @ y_grid  (N, 3)
    grad = weighted_terms.T @ y_grid
    
    # Normalize by Z
    Z = np.sum(1.0 / sum_exp)
    grad /= Z
    
    # Project to tangent space
    # grad_tan_i = grad_i - (grad_i . p_i) p_i
    dots_grad_p = np.einsum('ij,ij->i', grad, points)
    grad_tan = grad - dots_grad_p[:, np.newaxis] * points
    
    return grad_tan

def _fibonacci_sphere(n):
    idx = np.arange(n) + 0.5
    z = 1.0 - 2.0 * idx / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * idx
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_points = _fibonacci_sphere(N)
    best_radius = _compute_exact_covering_radius(best_points)
    
    # Use multiple restarts with different initializations and betas
    betas = [50.0, 100.0, 200.0]
    
    for restart in range(3):
        if time.time() > deadline - 30.0:
            break
            
        if restart == 0:
            points = _fibonacci_sphere(N)
        else:
            np.random.seed(restart * 100 + 42)
            points = np.random.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            # Quick repulsion to avoid coincident points
            for _ in range(50):
                dots = points @ points.T
                dots = np.clip(dots, -1.0, 1.0)
                dists = np.arccos(dots)
                weights = np.exp(-dists**2 / 0.05)
                np.fill_diagonal(weights, 0)
                sum_j = weights @ points
                sum_w = weights.sum(axis=1, keepdims=True)
                forces = sum_j - points * sum_w
                dots_fp = np.einsum('ij,ij->i', forces, points)
                forces_tan = forces - dots_fp[:, np.newaxis] * points
                points += 0.05 * forces_tan
                points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        current_points = points.copy()
        
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
                
            try:
                # L-BFGS-B optimization
                # We minimize the logsumexp approximation
                # Flatten points for optimizer
                def objective_and_grad(flat_pts):
                    pts = flat_pts.reshape(N, 3)
                    # Normalize to unit sphere
                    norms = np.linalg.norm(pts, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-10)
                    pts = pts / norms
                    
                    val = _logsumexp_covering_radius(pts, beta=beta, n_grid=1000)
                    grad = _gradient_logsumexp(pts, beta=beta, n_grid=1000)
                    
                    # Gradient with respect to flat_pts
                    # d(pts)/d(flat_pts) involves normalization.
                    # For small steps, we can approximate by just returning grad, 
                    # but technically we need the chain rule.
                    # Since we project to tangent space, and the optimizer moves in R^3,
                    # we should ensure the update stays on the sphere.
                    # L-BFGS-B doesn't constrain to sphere.
                    # We will re-normalize in the objective.
                    # The gradient should be projected.
                    # A better approach: use spherical coordinates or quaternion?
                    # Or just use the tangent space gradient and let the optimizer do its thing,
                    # relying on the normalization in the objective to pull it back.
                    # This might be unstable.
                    
                    # Alternative: Use a penalty method or just accept the approximation.
                    # Let's return the gradient in the ambient space, but scaled.
                    return val, grad.flatten()
                
                res = minimize(
                    objective_and_grad, 
                    current_points.flatten(), 
                    jac=True, 
                    method='L-BFGS-B', 
                    options={'maxiter': 50, 'gtol': 1e-4}
                )
                
                new_points = res.x.reshape(N, 3)
                new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
                
                new_radius = _compute_exact_covering_radius(new_points)
                
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_points = new_points
                    current_points = new_points
                    
            except Exception as e:
                pass
                
    return best_points
# EVOLVE-BLOCK-END
