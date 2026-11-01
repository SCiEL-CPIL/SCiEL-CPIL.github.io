"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _initial_fib():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius(pts):
    """Compute the exact covering radius using convex hull facets."""
    if len(pts) < 4:
        return 180.0
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    max_radius = 0.0
    for simplex in hull.simplices:
        v1 = pts[simplex[0]]
        v2 = pts[simplex[1]]
        v3 = pts[simplex[2]]
        
        e1 = v2 - v1
        e2 = v3 - v1
        n = np.cross(e1, e2)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-15:
            continue
        
        d = np.dot(n, v1)
        dist = abs(d) / n_norm
        
        if dist > 1.0:
            dist = 1.0
        elif dist < -1.0:
            dist = -1.0
        
        angle = np.arccos(min(max(dist, -1.0), 1.0))
        if angle > max_radius:
            max_radius = angle
    
    return np.degrees(max_radius)

def _soft_min_obj(x, beta):
    """
    Objective: soft-min of squared angular distances to the N points.
    x: shape (N, 3)
    Returns: scalar value and gradient (N, 3)
    
    For a test point y, the distance to nearest point i is arccos(y . x_i).
    We want to minimize the max over y of this distance.
    Approximate by minimizing the soft-min of (arccos(y . x_i))^2 over i,
    but that's hard. Instead, use a differentiable surrogate:
    
    For each y, let d_i(y) = (1 - y . x_i). Minimizing max_i d_i(y) is equivalent
    to minimizing the covering radius. We use a log-sum-exp style soft-max:
    f(y) = (1/beta) * log(sum_i exp(beta * d_i(y)))
    
    Then minimize max_y f(y) or an integral of f(y) over the sphere.
    For simplicity and speed, we minimize the sum of f(y) over a dense grid,
    or better, we use the fact that the worst point y is the circumcenter of
    a facet. 
    
    Alternative simpler approach: minimize the sum of squared distances from
    a dense grid to the nearest point, using a smooth min.
    """
    # We will actually minimize a smooth function that approximates the covering radius.
    # Let's use the maximum of the squared chordal distances from a grid, smoothed.
    # But to be fully differentiable and fast, let's use the "log-sum-exp" of squared distances
    # to the points, evaluated at the points themselves? No.
    
    # Let's use a standard trick: minimize the sum over a grid of the soft-min of distances.
    # But that requires storing the grid. Let's precompute a grid.
    
    # Actually, a better differentiable objective for sphere covering is to minimize
    # the potential energy or to maximize the minimum distance (packing), but we are covering.
    
    # Let's stick to the strategy: minimize the soft-max of the angular distances to the points
    # for a set of test points. To make it fully differentiable, we use the gradient of the
    # soft-max.
    
    # For efficiency, we'll use a smaller grid for the gradient descent.
    # Precomputed grid is better.
    
    pass # This function is a placeholder, we'll implement the real one below.

def _soft_covering_obj(x, grid, beta):
    """
    x: (N, 3) points
    grid: (M, 3) test points
    beta: sharpness of soft-min
    Returns: objective value and gradient
    """
    # Compute dot products: grid (M,3) @ x.T (3,N) -> (M,N)
    dots = grid @ x.T
    dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
    
    # Angular distances
    angles = np.arccos(dots)  # (M, N)
    
    # We want to minimize the max over M of the min over N of angles.
    # Soft-min over N: (1/beta) * log(sum(exp(-beta * angles)))
    # But log-sum-exp can overflow. Use the stable version.
    
    # For each row m, compute soft_min_n angles[m, n]
    # soft_min = -1/beta * log(sum(exp(-beta * angles)))
    
    # To avoid overflow, subtract max (or min) first.
    # Let z = -beta * angles. We want log(sum(exp(z))).
    # max_z = max(z, axis=1, keepdims=True)
    # log_sum_exp = max_z + log(sum(exp(z - max_z)))
    # soft_min = -1/beta * log_sum_exp
    
    z = -beta * angles
    max_z = np.max(z, axis=1, keepdims=True)
    # Handle case where all z are -inf (shouldn't happen with clip)
    exp_z = np.exp(z - max_z)
    sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
    log_sum_exp = max_z[:, 0] + np.log(sum_exp_z[:, 0])
    
    soft_min = -log_sum_exp / beta  # (M,)
    
    # Objective: max of soft_min, or sum? Sum is smoother. Let's use sum or max.
    # Max is more directly related to covering radius. But max is not differentiable.
    # Use a soft-max over M as well? That's getting complex.
    # Let's just minimize the mean of the soft-mins, or the max.
    # Let's use the max of the soft-mins, approximated by a soft-max.
    
    # Actually, minimizing the sum of squared soft-mins is a good differentiable surrogate.
    obj = np.mean(soft_min ** 2)
    
    # Gradient
    # d(obj)/d(angles[m,n]) = 2 * soft_min[m] / M * d(soft_min[m])/d(angles[m,n])
    # d(soft_min[m])/d(angles[m,n]) = -1/beta * d(log_sum_exp)/d(angles[m,n])
    # d(log_sum_exp)/d(z[m,n]) = exp(z[m,n]) / sum_exp_z[m]
    # d(z[m,n])/d(angles[m,n]) = -beta
    # So d(soft_min[m])/d(angles[m,n]) = -1/beta * (exp(z[m,n]) / sum_exp_z[m]) * (-beta) = exp(z[m,n]) / sum_exp_z[m]
    
    # Let p[m,n] = exp(z[m,n]) / sum_exp_z[m]  (this is the softmax weights)
    p = exp_z / sum_exp_z  # (M, N)
    
    # d(obj)/d(angles) = 2 * soft_min[:, None] * p / M
    d_obj_d_angles = 2.0 * soft_min[:, None] * p / len(grid)
    
    # Now, d(angles)/d(x)
    # angles = arccos(dots)
    # d(angles)/d(dots) = -1 / sqrt(1 - dots^2)
    # d(dots)/d(x) = grid[:, :, None]  (M, N, 3) where d(dots[m,n])/d(x[k,j]) = grid[m,j] if k==n else 0
    
    # This is tricky. Let's compute gradient with respect to x.
    # d(obj)/d(x[k,j]) = sum_m d_obj_d_angles[m,k] * d(angles[m,k])/d(dots[m,k]) * d(dots[m,k])/d(x[k,j])
    # d(angles[m,k])/d(dots[m,k]) = -1 / sqrt(1 - dots[m,k]^2)
    # d(dots[m,k])/d(x[k,j]) = grid[m,j]
    
    # So d(obj)/d(x[k,j]) = grid[m,j] * sum_m d_obj_d_angles[m,k] * (-1 / sqrt(1 - dots[m,k]^2))
    
    # Let's vectorize this.
    # denom = sqrt(1 - dots^2)  (M, N)
    denom = np.sqrt(np.maximum(1.0 - dots**2, 1e-12))  # (M, N)
    
    # d_obj_d_dots = d_obj_d_angles * (-1 / denom)  (M, N)
    d_obj_d_dots = -d_obj_d_angles / denom
    
    # Now, d_obj_d_x[k,j] = sum_m grid[m,j] * d_obj_d_dots[m,k]
    # This is a matrix multiplication: grid.T (3, M) @ d_obj_d_dots (M, N) -> (3, N)
    # Then transpose to get (N, 3)
    grad = (grid.T @ d_obj_d_dots).T  # (N, 3)
    
    return obj, grad

def _lloyd_iteration(pts, grid):
    """One Lloyd iteration: move each point to the centroid of its Voronoi cell."""
    dots = grid @ pts.T
    assignments = dots.argmax(axis=1)
    
    new_pts = pts.copy()
    for i in range(len(pts)):
        mask = assignments == i
        if np.sum(mask) > 0:
            centroid = grid[mask].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                new_pts[i] = centroid / norm
    
    return new_pts

def construct():
    """Construct 130 points on the unit sphere to minimize the covering radius."""
    start_time = time.time()
    max_time = 150.0  # Safety margin for 150s limit
    
    # Start with Fibonacci spiral
    pts = _initial_fib()
    
    # Precompute a grid for optimization
    n_theta = 100
    n_phi = 200
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    # Phase 1: Lloyd refinement
    current = best_pts.copy()
    for it in range(5):
        if time.time() - start_time > max_time - 60:
            break
        current = _lloyd_iteration(current, grid)
        # Normalize
        current /= np.linalg.norm(current, axis=1, keepdims=True)
        r = _covering_radius(current)
        if r < best_radius:
            best_radius = r
            best_pts = current.copy()
    
    # Phase 2: Gradient descent with soft-min objective
    if time.time() - start_time < max_time - 30:
        x0 = best_pts.reshape(-1)
        
        def obj_func(x):
            pts_reshaped = x.reshape(N, 3)
            # Normalize
            norms = np.linalg.norm(pts_reshaped, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            pts_reshaped = pts_reshaped / norms
            val, _ = _soft_covering_obj(pts_reshaped, grid, beta=10.0)
            return val
        
        def grad_func(x):
            pts_reshaped = x.reshape(N, 3)
            norms = np.linalg.norm(pts_reshaped, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            pts_reshaped = pts_reshaped / norms
            _, grad = _soft_covering_obj(pts_reshaped, grid, beta=10.0)
            # Chain rule for normalization
            # d(x/norm)/dx = (I - x x^T / norm^2) / norm
            # This is complex, so we'll just use the gradient as is, assuming x is normalized.
            # Or we can project the gradient.
            return grad.reshape(-1)
        
        try:
            res = minimize(obj_func, x0, method='L-BFGS-B', jac=grad_func,
                           options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
            if res.success or res.fun < obj_func(x0):
                new_pts = res.x.reshape(N, 3)
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
                new_radius = _covering_radius(new_pts)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_pts = new_pts
        except Exception:
            pass
    
    # Phase 3: Facet-based refinement
    if time.time() - start_time < max_time - 10:
        current = best_pts.copy()
        current_radius = best_radius
        
        for it in range(50):
            if time.time() - start_time > max_time - 5:
                break
            
            try:
                hull = ConvexHull(current)
            except Exception:
                break
            
            # Find the worst facet
            worst_idx = -1
            worst_angle = 0.0
            for si, simplex in enumerate(hull.simplices):
                v1 = current[simplex[0]]
                v2 = current[simplex[1]]
                v3 = current[simplex[2]]
                
                e1 = v2 - v1
                e2 = v3 - v1
                n = np.cross(e1, e2)
                n_norm = np.linalg.norm(n)
                
                if n_norm < 1e-15:
                    continue
                
                d = np.dot(n, v1)
                dist = abs(d) / n_norm
                dist = min(max(dist, -1.0), 1.0)
                angle = np.arccos(dist)
                
                if angle > worst_angle:
                    worst_angle = angle
                    worst_idx = si
            
            if worst_idx < 0:
                break
            
            simplex = hull.simplices[worst_idx]
            
            # Move the points of the worst facet towards the circumcenter
            # The circumcenter is in the direction of the normal
            v1 = current[simplex[0]]
            v2 = current[simplex[1]]
            v3 = current[simplex[2]]
            e1 = v2 - v1
            e2 = v3 - v1
            n = np.cross(e1, e2)
            n_norm = np.linalg.norm(n)
            
            if n_norm < 1e-15:
                break
            
            # Direction from origin to circumcenter
            # The circumcenter is at (1/dist) * n / n_norm, where dist = dot(n, v1)/n_norm
            d = np.dot(n, v1)
            dist = abs(d) / n_norm
            dist = min(max(dist, -1.0), 1.0)
            
            if dist < 1e-10:
                break
                
            # Circumcenter direction
            cc_dir = n / (n_norm * dist)
            
            # Move each point slightly towards the circumcenter
            for idx in simplex:
                # Move towards cc_dir
                step = 0.01
                new_pt = current[idx] + step * cc_dir
                new_pt = new_pt / np.linalg.norm(new_pt)
                current[idx] = new_pt
            
            new_radius = _covering_radius(current)
            if new_radius < current_radius:
                current_radius = new_radius
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_pts = current.copy()
            else:
                # Revert if it didn't help
                # We don't have a copy, so we'll just break or try smaller step
                # For simplicity, break
                break
    
    # Final normalization and check
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-10:
                best_pts[j] += np.random.randn(3) * 0.001
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    return best_pts
# EVOLVE-BLOCK-END
