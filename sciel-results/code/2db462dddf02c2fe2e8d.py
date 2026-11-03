"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import time

N = 50
SEED_DEADLINE = 145.0  # safety margin before 150s wall-clock limit


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute the covering radius (max angular distance to nearest point) in radians.
    
    Uses the convex hull approach: the covering radius is the maximum circumradius
    of the spherical Voronoi cells (facets of the convex hull).
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: brute force on a grid
        theta = np.linspace(0, np.pi, 100)
        phi = np.linspace(0, 2 * np.pi, 200)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        X = np.sin(TH) * np.cos(PH)
        Y = np.sin(TH) * np.sin(PH)
        Z = np.cos(TH)
        grid_pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        # Compute distances to each config point
        # For each grid point, find min angle to any config point
        # dot product with config points
        dots = grid_pts @ pts.T  # (n_grid, N)
        min_dots = dots.max(axis=1)  # max dot = min angle
        max_angle = np.arccos(np.clip(1 - min_dots, -1, 1))
        return np.max(max_angle)
    
    # For each facet (simplex) of the hull, compute the circumcenter
    # and the angular radius (max angle from circumcenter to facet vertices)
    max_radius = 0.0
    for simplex in hull.simplices:
        facet_pts = pts[simplex]
        # Compute circumcenter of the facet in 3D
        # For a simplex with vertices v0, v1, ..., vk, the circumcenter c
        # satisfies |c - vi|^2 = |c - v0|^2 for all i
        # This gives linear equations: 2*(vi - v0) . c = |vi|^2 - |v0|^2
        v0 = facet_pts[0]
        A = 2.0 * (facet_pts[1:] - v0)  # (k, 3)
        b = np.sum(facet_pts[1:]**2, axis=1) - np.sum(v0**2)  # (k,)
        try:
            c = np.linalg.lstsq(A, b, rcond=None)[0]
        except Exception:
            continue
        # The circumcenter on the sphere is c normalized
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            continue
        c_unit = c / c_norm
        # The angular radius is the angle from c_unit to any vertex
        # But we need the "empty cap" check: the cap centered at c_unit
        # with angular radius alpha should contain no other points
        # alpha = arccos(c_unit . v0)
        dot_val = np.clip(np.dot(c_unit, v0), -1, 1)
        alpha = np.arccos(dot_val)
        
        # Check that no other point is closer to c_unit than the facet vertices
        # i.e., all other points should be outside the cap
        dots_all = c_unit @ pts.T
        # The facet vertices should have the largest dots (smallest angles)
        # If any other point has a larger dot, the cap is not empty
        facet_dots = dots_all[simplex]
        min_facet_dot = np.min(facet_dots)
        max_other_dot = np.max(np.delete(dots_all, simplex))
        
        if max_other_dot <= min_facet_dot + 1e-9:
            # Cap is empty (or nearly so)
            max_radius = max(max_radius, alpha)
        else:
            # Cap is not empty, this facet doesn't determine the covering radius
            # The actual covering radius for this region is determined by
            # the nearest point to the circumcenter
            nearest_dot = np.max(dots_all)
            angle_to_nearest = np.arccos(np.clip(nearest_dot, -1, 1))
            max_radius = max(max_radius, angle_to_nearest)
    
    return max_radius


def _fibonacci_points(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _gradient_descent(pts, max_iter=2000, lr=0.01, deadline=None):
    """Perform gradient descent on the covering radius objective.
    
    Uses a smooth approximation: minimize the log-sum-exp of squared angular distances
    from a dense set of sample points to the nearest configuration point.
    """
    pts = _normalize(pts.copy())
    
    # Sample points on the sphere for the objective
    n_samples = 2000
    # Use Fibonacci sampling for the test points
    i = np.arange(n_samples) + 0.5
    z_s = 1.0 - 2.0 * i / n_samples
    phi_s = np.pi * (1.0 + np.sqrt(5.0)) * i
    r_s = np.sqrt(1.0 - z_s * z_s)
    sample_pts = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute distances from each sample to each config point
        # dot products
        dots = sample_pts @ pts.T  # (n_samples, N)
        # For each sample, the nearest point gives the max dot
        # We want to minimize the max angular distance = minimize -min(log(1-dot)) or similar
        
        # Smooth max via log-sum-exp
        # Let f_i = -log(1 - dot_i) for each config point i
        # The covering radius at sample j is max_i f_ji
        # Use soft-max: log(sum(exp(f_ji)))
        
        # Clip dots to avoid log(0)
        dots_clipped = np.clip(dots, -1 + 1e-10, 1 - 1e-10)
        f = -np.log(1 - dots_clipped)  # (n_samples, N)
        
        # Soft max over config points for each sample
        f_max = np.max(f, axis=1)
        f_shifted = f - f_max[:, np.newaxis]
        soft_max = f_max + np.log(np.sum(np.exp(f_shifted), axis=1))
        
        # Objective: mean of soft_max values
        obj = np.mean(soft_max)
        
        # Gradient with respect to pts
        # d(soft_max_j)/d(pts_k) = sum_i exp(f_ji - soft_max_j) * d(f_ji)/d(pts_k)
        # d(f_ji)/d(pts_k) = -d(log(1 - dot_ji))/d(pts_k) = -(-1/(1-dot_ji)) * d(dot_ji)/d(pts_k)
        # = 1/(1-dot_ji) * sample_j (since dot_ji = sample_j . pts_i, and if i==k)
        
        # Weights for each (sample, config) pair
        weights = np.exp(f_shifted) / np.sum(np.exp(f_shifted), axis=1, keepdims=True)  # (n_samples, N)
        
        # Gradient of obj w.r.t. pts[k]
        # d_obj/d_pts_k = (1/n_samples) * sum_j weights[j,k] * sample_j / (1 - dots[j,k])
        grad = np.zeros_like(pts)
        for k in range(N):
            denom = 1 - dots[:, k]
            denom = np.clip(denom, 1e-10, None)
            contrib = weights[:, k] / denom  # (n_samples,)
            grad[k] = np.mean(contrib[:, np.newaxis] * sample_pts, axis=0)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad = grad / grad_norm
        
        # Update
        pts = pts - lr * grad
        pts = _normalize(pts)
        
        # Adaptive learning rate
        if it % 100 == 0:
            lr = max(0.001, lr * 0.9)
    
    return pts


def _local_search(pts, n_restarts=5, max_iter=500, deadline=None):
    """Local search: perturb points and keep improvements."""
    best_pts = _normalize(pts.copy())
    best_radius = _covering_radius(best_pts)
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        pts = best_pts.copy()
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            # Pick a random point to move
            idx = np.random.randint(N)
            # Generate a random direction
            new_dir = np.random.randn(3)
            new_dir /= np.linalg.norm(new_dir)
            # Small perturbation
            delta = 0.05 * new_dir
            old_pt = pts[idx].copy()
            pts[idx] = old_pt + delta
            pts[idx] = pts[idx] / np.linalg.norm(pts[idx])
            
            new_radius = _covering_radius(pts)
            if new_radius < best_radius:
                best_radius = new_radius
                best_pts = pts.copy()
            else:
                pts[idx] = old_pt
    
    return best_pts, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + SEED_DEADLINE
    
    # Start with Fibonacci spiral
    pts = _fibonacci_points(N)
    
    # Gradient descent refinement
    pts = _gradient_descent(pts, max_iter=1000, lr=0.02, deadline=deadline)
    
    # Local search
    np.random.seed(42)
    pts, radius = _local_search(pts, n_restarts=3, max_iter=200, deadline=deadline)
    
    # Final gradient polish
    if time.time() < deadline:
        pts = _gradient_descent(pts, max_iter=500, lr=0.005, deadline=deadline)
    
    return _normalize(pts)
