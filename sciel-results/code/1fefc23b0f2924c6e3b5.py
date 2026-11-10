import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _facet_data(points):
    """Compute facet normals, distances, and circumcenters (Voronoi vertices).
    Returns (simplices, normals, norms, d, alphas, circumcenters) or None on failure."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return None
    simplices = hull.simplices
    tri_pts = points[simplices]
    p0 = tri_pts[:, 0, :]
    p1 = tri_pts[:, 1, :]
    p2 = tri_pts[:, 2, :]
    v1 = p1 - p0
    v2 = p2 - p0
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)
    mask = norms > 1e-12
    if not np.any(mask):
        return None
    d = np.zeros(len(normals))
    d[mask] = np.abs(np.sum(normals[mask] * p0[mask], axis=1)) / norms[mask]
    d = np.clip(d, 0.0, 1.0)
    alphas = np.arccos(d)
    # Circumcenter: normalized centroid of the triangle (lies on the normal line through origin)
    centroids = (p0 + p1 + p2) / 3.0
    c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    c_norms[c_norms < 1e-12] = 1.0
    circumcenters = centroids / c_norms
    return simplices, normals, norms, d, alphas, circumcenters

def _spherical_kmeans_init(n_clusters=50, n_iter=30, n_restarts=5, seed=42):
    """Spherical K-Means for better initial configuration."""
    rng = np.random.RandomState(seed)
    best_points = None
    best_cost = np.inf
    
    for restart in range(n_restarts):
        centers = rng.randn(n_clusters, 3)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        
        for _ in range(n_iter):
            # Proper spherical k-means: assign to nearest center, recompute centers
            # We use a set of "sample" points on the sphere for assignment
            # For initialization purposes, just randomize and normalize
            pass
        
        points = centers.copy()
        cost = _covering_radius(points)
        if cost < best_cost:
            best_cost = cost
            best_points = points.copy()
    
    return best_points

def _covering_radius(points):
    """Calculate the covering radius in radians for points on the unit sphere."""
    fd = _facet_data(points)
    if fd is None:
        return np.pi
    _, _, _, _, alphas, _ = fd
    return np.max(alphas)

def _covering_radius_smooth(points, beta=50.0):
    """Smooth surrogate for the covering radius using log-sum-exp."""
    fd = _facet_data(points)
    if fd is None:
        return np.pi
    _, _, _, _, alphas, _ = fd
    max_alpha = np.max(alphas)
    shifted = alphas - max_alpha
    log_sum_exp = max_alpha + np.log(np.sum(np.exp(shifted)))
    return log_sum_exp / beta

def _voronoi_smooth_obj_grad(params, beta=50.0):
    """
    Smooth objective and analytic gradient based on Voronoi vertices (circumcenters).
    
    The covering radius equals the max over Voronoi vertices v_k of the angular
    distance from v_k to its nearest generating point. We use a smooth max
    (log-sum-exp) surrogate and compute the gradient analytically.
    
    For a Voronoi vertex v (unit vector, circumcenter of triangle with vertices
    p_i, p_j, p_k), the angular distance to p_i is:
        theta_i = arccos(v . p_i)
    
    d theta_i / d p_i = -v / sqrt(1 - (v.p_i)^2)  (projected onto tangent plane)
    
    We use the smooth max: f = (1/beta) * log(sum_k exp(beta * theta_k))
    where theta_k is the angular distance from Voronoi vertex k to its nearest point.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    if not np.all(np.isfinite(pts)):
        return 1e6, np.zeros_like(params)
    
    fd = _facet_data(pts)
    if fd is None:
        return 1e6, np.zeros_like(params)
    
    simplices, normals, norms, d, alphas, circumcenters = fd
    
    # For each Voronoi vertex (circumcenter), find the angular distance to each
    # of its three neighboring points. The nearest one determines the local covering.
    # We use a smooth min over the three angular distances per vertex.
    
    tri_pts = pts[simplices]  # (F, 3, 3)
    # v . p_i for each vertex and each of its 3 neighbors
    dots = np.sum(circumcenters[:, np.newaxis, :] * tri_pts, axis=2)  # (F, 3)
    dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
    
    # Angular distances
    thetas = np.arccos(dots)  # (F, 3)
    
    # For each vertex, the relevant angular distance is the minimum (nearest point)
    # But for the covering radius, we want the max over all vertices of (min distance to nearest)
    # Actually, the covering radius at vertex v is the distance to the NEAREST point.
    # So local_radius_k = min_i theta_{k,i}
    
    # Smooth min: -logsumexp(-beta * thetas) / beta
    # Then smooth max over vertices: logsumexp(beta * local_radius) / beta
    
    # Let's use: for each vertex, compute smooth min of its 3 thetas
    neg_thetas = -thetas
    max_neg = np.max(neg_thetas, axis=1, keepdims=True)
    exp_shifted = np.exp(neg_thetas - max_neg)
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
    log_sum_exp_neg = max_neg + np.log(sum_exp)
    smooth_min_thetas = -log_sum_exp_neg.squeeze(1)  # (F,)
    
    # Now smooth max over all vertices
    max_sm = np.max(smooth_min_thetas)
    shifted_sm = smooth_min_thetas - max_sm
    exp_sm = np.exp(shifted_sm)
    log_sum_exp_sm = max_sm + np.log(np.sum(exp_sm))
    f = log_sum_exp_sm / beta
    
    # Gradient computation
    # d f / d p_i via chain rule
    grad = np.zeros_like(pts)
    
    # d log_sum_exp_sm / d smooth_min_thetas_k = exp_sm[k] / sum(exp_sm)
    weights_vertex = exp_sm / np.sum(exp_sm)  # (F,)
    
    # d smooth_min_thetas_k / d theta_{k,i} = exp_shifted[k,i] / sum_exp[k]
    # (from the soft-min formula)
    weights_theta = exp_shifted.squeeze(1) / sum_exp.squeeze(1)  # (F, 3)
    
    # d theta_{k,i} / d p_j where p_j is the i-th vertex of triangle k
    # theta = arccos(v . p), d theta/dp = -v / sqrt(1 - (v.p)^2)
    # But we need to project onto the tangent plane (since p is constrained to sphere)
    # The unconstrained gradient projected: grad_tangent = grad - (grad.p)*p
    
    for k in range(len(simplices)):
        wk = weights_vertex[k]
        if wk < 1e-15:
            continue
        v = circumcenters[k]
        for i in range(3):
            wi = weights_theta[k, i]
            if wi < 1e-15:
                continue
            p = tri_pts[k, i]
            dot_val = dots[k, i]
            # d theta / d p = -v / sqrt(1 - dot^2)
            denom = np.sqrt(np.maximum(1e-12, 1.0 - dot_val * dot_val))
            grad_unconstrained = -v / denom
            # Project onto tangent plane
            proj = grad_unconstrained - np.dot(grad_unconstrained, p) * p
            idx = simplices[k, i]
            grad[idx] += wk * wi * proj
    
    # Scale by 1/beta
    grad /= beta
    
    # Also account for the normalization of pts (params -> pts via normalize)
    # pts = params / |params|
    # d pts / d params_j = (I - pts*pts^T) / |params|
    # So grad_params = (I - pts*pts^T) @ grad_pts / |params|
    norms_params = np.linalg.norm(params.reshape(N, 3), axis=1, keepdims=True)
    norms_params[norms_params < 1e-12] = 1.0
    # Projection: grad_tangent = grad - (grad . pts) * pts
    grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
    grad_params = grad_tangent / norms_params
    
    return f, grad_params.flatten()

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms

def _voronoi_optimize(points, beta=30.0, max_iter=200, time_limit=None, start_time=None):
    """
    Optimize using L-BFGS with analytic gradient from Voronoi vertex formulation.
    """
    if start_time is None:
        start_time = time.time()
    
    def obj_grad(params):
        if time_limit is not None and time.time() - start_time > time_limit:
            return 1e6, np.zeros_like(params)
        return _voronoi_smooth_obj_grad(params, beta=beta)
    
    params0 = points.flatten()
    try:
        result = minimize(obj_grad, params0, method='L-BFGS-B', jac=True,
                          options={'maxiter': max_iter, 'gtol': 1e-8, 'disp': False,
                                   'maxfun': max_iter * 3})
        return _normalize(result.x.reshape(N, 3))
    except Exception:
        return points

def _repulsive_relaxation(points, steps=500, alpha=0.1, time_limit=None, start_time=None):
    """
    Relax points using repulsive forces to spread them out.
    """
    if start_time is None:
        start_time = time.time()
    points = points.copy()
    n = points.shape[0]
    
    for step in range(steps):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        # Compute all pairwise differences (vectorized)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        mask = dist_sq > 1e-12
        inv_dist3 = np.zeros_like(dist_sq)
        inv_dist3[mask] = 1.0 / (dist_sq[mask] * np.sqrt(dist_sq[mask]))
        
        forces = np.sum(diff * inv_dist3[:, :, np.newaxis], axis=1)
        
        points += alpha * forces
        points = _normalize(points)
        
    return points

def _covering_force(points):
    """
    Compute forces that push points toward regions of high covering radius.
    Uses Voronoi vertex formulation.
    """
    n = points.shape[0]
    fd = _facet_data(points)
    if fd is None:
        # Fallback: uniform repulsion
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        mask = dist_sq > 1e-12
        inv_dist3 = np.zeros_like(dist_sq)
        inv_dist3[mask] = 1.0 / (dist_sq[mask] * np.sqrt(dist_sq[mask]))
        forces = np.sum(diff * inv_dist3[:, :, np.newaxis], axis=1)
        return forces
    
    simplices, normals, norms, d, alphas, circumcenters = fd
    
    # For each facet, the Voronoi vertex (circumcenter) is the worst point in that cell.
    # Push the three vertices of the triangle toward the circumcenter, weighted by alpha.
    tri_pts = points[simplices]
    forces = np.zeros_like(points)
    
    max_alpha = np.max(alphas)
    # Weight by how much larger this facet's radius is
    weights = np.exp(20 * (alphas - max_alpha))
    
    for k in range(len(simplices)):
        if weights[k] < 1e-15:
            continue
        v = circumcenters[k]
        for i in range(3):
            idx = simplices[k, i]
            p = tri_pts[k, i]
            # Direction from p toward v, projected onto tangent plane
            direction = v - np.dot(v, p) * p
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-12:
                forces[idx] += weights[k] * direction / norm_dir
    
    return forces

def _circumcenter_polish(points, max_iter=200, tol=1e-10, time_limit=None, start_time=None):
    """
    Polish by moving each point toward the circumcenter of its largest adjacent facet.
    This directly reduces the covering radius.
    """
    if start_time is None:
        start_time = time.time()
    points = points.copy()
    n = points.shape[0]
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        fd = _facet_data(points)
        if fd is None:
            break
        
        simplices, normals, norms, d, alphas, circumcenters = fd
        
        # For each point, find its largest adjacent facet and move toward its circumcenter
        updates = np.zeros_like(points)
        counts = np.zeros(n)
        
        max_alpha = np.max(alphas)
        
        for i, simplex in enumerate(simplices):
            weight = np.exp(50 * (alphas[i] - max_alpha))
            if weight < 1e-15:
                continue
            cc = circumcenters[i]
            for vertex_idx in simplex:
                updates[vertex_idx] += weight * (cc - points[vertex_idx])
                counts[vertex_idx] += weight
        
        # Apply updates with damping
        for i in range(n):
            if counts[i] > 1e-10:
                updates[i] /= counts[i]
        
        # Damped update
        step_size = 0.1
        points = points + step_size * updates
        points = _normalize(points)
        
        # Check convergence
        if np.max(np.abs(updates)) < tol:
            break
    
    return points

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    def time_remaining():
        return deadline - time.time()

    # 1. Initial Configuration: Try both Fibonacci and K-Means
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_points = _normalize(fib_points)
    
    km_points = _spherical_kmeans_init(n_clusters=N, n_iter=20, n_restarts=3, seed=42)
    
    best_points = fib_points.copy()
    best_radius = _covering_radius(best_points)
    
    if _covering_radius(km_points) < best_radius:
        best_points = km_points
        best_radius = _covering_radius(best_points)
    
    # Relax the initial configuration
    relaxed = _repulsive_relaxation(best_points, steps=800, alpha=0.05, time_limit=4, start_time=start_time)
    rad = _covering_radius(relaxed)
    if rad < best_radius:
        best_points = relaxed
        best_radius = rad

    # 2. Define Objective for Optimizer
    def objective(params):
        if time.time() > deadline:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        if not np.all(np.isfinite(pts)):
            return 1e6
        return _covering_radius(pts)

    def objective_smooth(params):
        if time.time() > deadline:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        if not np.all(np.isfinite(pts)):
            return 1e6
        return _covering_radius_smooth(pts, beta=50.0)

    def optimize_from(points):
        nonlocal best_points, best_radius
        
        # Phase 1: L-BFGS with Voronoi vertex analytic gradient
        if time_remaining() > 15:
            voronoi_result = _voronoi_optimize(points, beta=30.0, max_iter=150,
                                                time_limit=12, start_time=start_time)
            rad_v = _covering_radius(voronoi_result)
            if rad_v < best_radius:
                best_points = voronoi_result
                best_radius = rad_v
        
        # Phase 2: Circumcenter polish
        if time_remaining() > 10:
            polished = _circumcenter_polish(best_points, max_iter=100, time_limit=5, start_time=start_time)
            rad = _covering_radius(polished)
            if rad < best_radius:
                best_points = polished
                best_radius = rad
        
        # Phase 3: Nelder-Mead polish with exact objective
        if time_remaining() > 10:
            try:
                result_nm = minimize(objective, best_points.flatten(), method='Nelder-Mead',
                                     options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if result_nm.fun < best_radius:
                    best_points = _normalize(result_nm.x.reshape(N, 3))
                    best_radius = result_nm.fun
            except Exception:
                pass

    # Run optimization from the best relaxed point
    optimize_from(best_points)
    
    # 4. Random Restarts
    while time_remaining() > 20:
        # Perturb current best
        noise_scale = 0.25
        noise = np.random.normal(0, noise_scale, size=best_points.shape)
        perturbed = best_points + noise
        perturbed = _normalize(perturbed)
        
        # Quick relaxation
        relaxed = _repulsive_relaxation(perturbed, steps=150, alpha=0.05, 
                                        time_limit=2, start_time=start_time)
        rad = _covering_radius(relaxed)
        
        if rad < best_radius:
            best_points = relaxed
            best_radius = rad
            optimize_from(best_points)
        elif time_remaining() > 30:
            # Even if not better, try Voronoi optimization from perturbed
            voronoi_result = _voronoi_optimize(perturbed, beta=40.0, max_iter=80,
                                                time_limit=5, start_time=start_time)
            rad2 = _covering_radius(voronoi_result)
            if rad2 < best_radius:
                best_points = voronoi_result
                best_radius = rad2
                optimize_from(best_points)

    # 5. Final Polish
    if time_remaining() > 10:
        # Voronoi L-BFGS with tighter beta
        voronoi_final = _voronoi_optimize(best_points, beta=60.0, max_iter=100,
                                           time_limit=6, start_time=start_time)
        rad_vf = _covering_radius(voronoi_final)
        if rad_vf < best_radius:
            best_points = voronoi_final
            best_radius = rad_vf
        
        # Circumcenter polish
        if time_remaining() > 6:
            polished = _circumcenter_polish(best_points, max_iter=200, time_limit=4, start_time=start_time)
            rad = _covering_radius(polished)
            if rad < best_radius:
                best_points = polished
                best_radius = rad
        
        # Then Nelder-Mead with tight tolerance
        if time_remaining() > 4:
            try:
                result_final = minimize(objective, best_points.flatten(), method='Nelder-Mead',
                                        options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                if result_final.fun < best_radius:
                    best_points = _normalize(result_final.x.reshape(N, 3))
                    best_radius = result_final.fun
            except Exception:
                pass

    return best_points
