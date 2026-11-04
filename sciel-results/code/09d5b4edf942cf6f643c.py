import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def compute_facet_data(points):
    """
    Computes the circumradius and circumcenter direction for each facet of the convex hull.
    Returns:
        radii: array of shape (M,)
        centers: array of shape (M, 3) unit vectors
        simplices: array of shape (M, 3)
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.array([]), np.array([]), np.array([])
    
    if len(simplices) == 0:
        return np.array([]), np.array([]), np.array([])
        
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    M = len(simplices)
    radii = np.full(M, np.pi)
    centers = np.zeros((M, 3))
    centers[:] = [0, 0, 1]
    
    valid = norms > 1e-10
    if not np.all(valid):
        if np.any(valid):
            a_v = a[valid]
            b_v = b[valid]
            c_v = c[valid]
            n_v = n[valid]
            norms_v = norms[valid]
            
            u = n_v / norms_v[:, np.newaxis]
            centroids = (a_v + b_v + c_v) / 3.0
            dots = np.sum(u * centroids, axis=1)
            u[dots < 0] *= -1
            
            cos_theta = np.sum(u * a_v, axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            radii[valid] = np.arccos(cos_theta)
            centers[valid] = u
        return radii, centers, simplices
    
    u = n / norms[:, np.newaxis]
    
    # Orient normals outwards
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return radii, u, simplices

def get_covering_radius(points):
    """Compute the covering radius in radians."""
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return np.pi
    return np.max(radii)

def greedy_init(n_points, seed=0):
    """
    Greedy initialization: start with a random point, then repeatedly add the point
    that maximizes the minimum distance to existing points.
    """
    np.random.seed(seed)
    points = [np.random.randn(3)]
    points[0] /= np.linalg.norm(points[0])
    
    for _ in range(n_points - 1):
        # Generate many candidates
        candidates = np.random.randn(1000, 3)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # For each candidate, compute min distance to existing points
        existing = np.array(points)
        # Dot products
        dots = candidates @ existing.T
        min_cos = np.min(dots, axis=1)
        min_cos = np.clip(min_cos, -1.0, 1.0)
        max_dist = np.arccos(min_cos)
        
        best_idx = np.argmax(max_dist)
        points.append(candidates[best_idx])
    
    return np.array(points)

def lloyd_step(points, max_iter=100, tol=1e-6):
    """
    Lloyd-like iteration: for each point, move it towards the centroid of its
    Voronoi cell (approximated by the centroid of neighboring hull facets).
    """
    points = points.copy()
    for it in range(max_iter):
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break
        
        # For each point, collect the centers of facets it belongs to
        # and the radii of those facets
        point_updates = np.zeros_like(points)
        point_counts = np.zeros(N)
        
        for m in range(len(radii)):
            simplex = simplices[m]
            i_idx, j_idx, k_idx = simplex
            center = centers[m]
            r = radii[m]
            
            # Weight by how "large" this facet is (larger radius = more urgent)
            w = r / np.max(radii) if np.max(radii) > 0 else 1.0
            
            for idx in [i_idx, j_idx, k_idx]:
                point_updates[idx] += w * center
                point_counts[idx] += w
        
        # Average the updates
        for i in range(N):
            if point_counts[i] > 0:
                point_updates[i] /= point_counts[i]
                # Project to tangent plane
                proj = np.dot(point_updates[i], points[i])
                point_updates[i] -= proj * points[i]
        
        # Apply small step
        step = 0.1
        points_new = points + step * point_updates
        points_new /= np.linalg.norm(points_new, axis=1, keepdims=True)
        
        # Check convergence
        if np.max(np.linalg.norm(points_new - points, axis=1)) < tol:
            points = points_new
            break
        points = points_new
    
    return points

def facet_push_step(points, max_iter=200, tol=1e-7):
    """
    Push vertices of the worst facet towards its circumcenter.
    """
    points = points.copy()
    for it in range(max_iter):
        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break
        
        max_idx = np.argmax(radii)
        max_r = radii[max_idx]
        
        if max_r < tol:
            break
        
        simplex = simplices[max_idx]
        i_idx, j_idx, k_idx = simplex
        center = centers[max_idx]
        
        # Move each vertex towards the center (tangentially)
        for idx in [i_idx, j_idx, k_idx]:
            p = points[idx]
            radial = np.dot(center - p, p)
            tangent_dir = (center - p) - radial * p
            norm_td = np.linalg.norm(tangent_dir)
            if norm_td > 1e-10:
                tangent_dir /= norm_td
                points[idx] += 0.01 * tangent_dir
        
        points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    return points

def smooth_objective_and_grad(x, tau=0.1):
    """
    Smooth objective: log-sum-exp of facet radii / tau.
    x is a flat array of shape (3*N,) representing N unit vectors.
    """
    points = x.reshape(N, 3)
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    points = points / norms
    
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return 1e10, np.zeros_like(x)
    
    # Log-sum-exp
    r_scaled = radii / tau
    r_max = np.max(r_scaled)
    exp_terms = np.exp(r_scaled - r_max)
    sum_exp = np.sum(exp_terms)
    obj = r_max + np.log(sum_exp)
    
    # Weights
    weights = exp_terms / sum_exp
    
    # Gradient
    grad_points = np.zeros_like(points)
    sin_r = np.sin(radii)
    sin_r[sin_r < 1e-6] = 1e-6
    
    for m in range(len(radii)):
        simplex = simplices[m]
        i_idx, j_idx, k_idx = simplex
        center = centers[m]
        w = weights[m]
        factor = w / (tau * sin_r[m])
        
        grad_points[i_idx] -= factor * center
        grad_points[j_idx] -= factor * center
        grad_points[k_idx] -= factor * center
    
    # Project to tangent plane
    for i in range(N):
        proj = np.dot(grad_points[i], points[i])
        grad_points[i] -= proj * points[i]
    
    grad = grad_points.reshape(-1)
    return obj, grad

def polish(points, time_limit=None, start_time=None):
    """
    Multi-phase polishing: L-BFGS-B with smooth surrogate, then worst-facet refinement.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 145.0
    
    best_points = points.copy()
    best_radius = get_covering_radius(points)
    
    # Phase 1: L-BFGS-B with smooth max surrogate
    x0 = points.reshape(-1).copy()
    tau = 0.1
    
    for tau_val in [0.1, 0.05, 0.02, 0.01]:
        if time.time() - start_time > time_limit - 10.0:
            break
        
        tau = tau_val
        res = minimize(
            smooth_objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000}
        )
        
        x_opt = res.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        r_opt = get_covering_radius(x_opt)
        if r_opt < best_radius:
            best_radius = r_opt
            best_points = x_opt.copy()
        
        x0 = x_opt.copy()
    
    # Phase 2: Worst-facet refinement
    if time.time() - start_time < time_limit - 5.0:
        points = best_points.copy()
        for step in range(500):
            if time.time() - start_time > time_limit - 2.0:
                break
            
            radii, centers, simplices = compute_facet_data(points)
            if len(radii) == 0:
                break
            
            max_idx = np.argmax(radii)
            max_r = radii[max_idx]
            
            if max_r < best_radius:
                best_radius = max_r
                best_points = points.copy()
            
            simplex = simplices[max_idx]
            i_idx, j_idx, k_idx = simplex
            center = centers[max_idx]
            
            updates = np.zeros_like(points)
            step_size = 0.005
            
            for idx in [i_idx, j_idx, k_idx]:
                p = points[idx]
                radial = np.dot(center - p, p)
                tangent_dir = (center - p) - radial * p
                norm_td = np.linalg.norm(tangent_dir)
                if norm_td > 1e-10:
                    tangent_dir /= norm_td
                updates[idx] += step_size * tangent_dir
            
            points += updates
            points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    return best_points, best_radius

def nelder_mead_refine(points, time_limit=None, start_time=None):
    """
    Nelder-Mead refinement on a subset of points near the worst facet.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 145.0
    
    best_points = points.copy()
    best_radius = get_covering_radius(points)
    
    # Find the worst facet and refine its vertices
    radii, centers, simplices = compute_facet_data(points)
    if len(radii) == 0:
        return best_points, best_radius
    
    max_idx = np.argmax(radii)
    simplex = simplices[max_idx]
    i_idx, j_idx, k_idx = simplex
    
    # Define objective for the 3 vertices
    def obj(x):
        p = best_points.copy()
        p[i_idx] = x[0:3]
        p[j_idx] = x[3:6]
        p[k_idx] = x[6:9]
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        return get_covering_radius(p)
    
    x0 = np.concatenate([best_points[i_idx], best_points[j_idx], best_points[k_idx]])
    
    res = minimize(
        obj,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
    )
    
    if time.time() - start_time < time_limit - 1.0:
        p = best_points.copy()
        p[i_idx] = res.x[0:3]
        p[j_idx] = res.x[3:6]
        p[k_idx] = res.x[6:9]
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        
        r = get_covering_radius(p)
        if r < best_radius:
            best_radius = r
            best_points = p.copy()
    
    return best_points, best_radius

def spectral_refinement(points, time_limit=None, start_time=None):
    """
    Spectral refinement: use the eigenstructure of the covariance matrix to
    identify clusters and refine them separately.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 145.0
    
    best_points = points.copy()
    best_radius = get_covering_radius(points)
    
    # Compute pairwise distances
    dots = points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # Find the 5 closest pairs and refine them
    idx = np.argsort(dists, axis=1)
    for i in range(N):
        if time.time() - start_time > time_limit - 5.0:
            break
        # Skip self
        nearest = idx[i, 1]
        
        # Refine the pair
        p = best_points.copy()
        # Small perturbation
        p[i] += 0.01 * np.random.randn(3)
        p[nearest] += 0.01 * np.random.randn(3)
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        
        r = get_covering_radius(p)
        if r < best_radius:
            best_radius = r
            best_points = p.copy()
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    best_points = None
    best_radius = np.pi
    
    # Initial configurations
    initializations = []
    
    # 1. Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(init_fib)
    
    # 2. Random points
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand /= np.linalg.norm(init_rand, axis=1, keepdims=True)
    initializations.append(init_rand)
    
    # 3. Greedy initialization
    init_greedy = greedy_init(N, seed=42)
    initializations.append(init_greedy)
    
    # 4. Another greedy with different seed
    init_greedy2 = greedy_init(N, seed=123)
    initializations.append(init_greedy2)
    
    # 5. Fibonacci + noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.2
    init_fib_noise /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
    initializations.append(init_fib_noise)
    
    # More randomized restarts
    for s in range(10):
        np.random.seed(1000 + s * 7)
        init = init_fib + np.random.randn(N, 3) * (0.1 + 0.1 * s / 10)
        init /= np.linalg.norm(init, axis=1, keepdims=True)
        initializations.append(init)
    
    max_restarts = min(len(initializations), 15)
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit - 20.0:
            break
        
        points = initializations[restart].copy()
        
        # Phase 1: Lloyd step
        points = lloyd_step(points, max_iter=50)
        
        # Phase 2: Facet push step
        points = facet_push_step(points, max_iter=100)
        
        # Phase 3: Multi-phase polishing
        points, radius = polish(points, time_limit=time_limit, start_time=start_time)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
        
        # Phase 4: Nelder-Mead refinement
        if time.time() - start_time < time_limit - 10.0:
            points, radius = nelder_mead_refine(points, time_limit=time_limit, start_time=start_time)
            if radius < best_radius:
                best_radius = radius
                best_points = points.copy()
        
        # Phase 5: Spectral refinement
        if time.time() - start_time < time_limit - 5.0:
            points, radius = spectral_refinement(points, time_limit=time_limit, start_time=start_time)
            if radius < best_radius:
                best_radius = radius
                best_points = points.copy()
    
    if best_points is None:
        best_points = init_fib
    
    # Final high-precision polish: focus only on the single worst facet
    if time.time() - start_time < time_limit - 1.0:
        final_points = best_points.copy()
        for step in range(500):
            if time.time() - start_time > time_limit - 0.5:
                break
            
            radii, centers, simplices = compute_facet_data(final_points)
            if len(radii) == 0:
                break
            
            max_idx = np.argmax(radii)
            max_r = radii[max_idx]
            
            if max_r < best_radius:
                best_radius = max_r
                best_points = final_points.copy()
            
            # Only move vertices of the worst facet
            simplex = simplices[max_idx]
            i_idx, j_idx, k_idx = simplex
            center = centers[max_idx]
            
            updates = np.zeros_like(final_points)
            step_size = 0.005
            
            for idx in [i_idx, j_idx, k_idx]:
                p = final_points[idx]
                radial = np.dot(center - p, p)
                tangent_dir = (center - p) - radial * p
                norm_td = np.linalg.norm(tangent_dir)
                if norm_td > 1e-10:
                    tangent_dir /= norm_td
                updates[idx] += step_size * tangent_dir
            
            final_points += updates
            final_points /= np.linalg.norm(final_points, axis=1, keepdims=True)
    
    return best_points
