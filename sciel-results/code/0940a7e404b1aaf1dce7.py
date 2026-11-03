import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius_rad(points):
    """Compute covering radius in radians via convex hull facets."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    simplices = hull.simplices
    if len(simplices) == 0:
        return np.pi
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    valid = norms > 1e-10
    if not np.all(valid):
        return np.pi
    u = n / norms[:, np.newaxis]
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    return np.max(radii)

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
    
    valid = norms > 1e-10
    if not np.all(valid):
        # Degenerate facets, handle gracefully
        radii = np.full(len(simplices), np.pi)
        centers = np.zeros((len(simplices), 3))
        centers[~valid] = [0, 0, 1]
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
        candidates = np.random.randn(500, 3)
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

def polish(points, time_budget):
    """
    High-precision polish pass with tighter tolerance and adaptive step size.
    """
    start = time.time()
    alpha = 0.01
    best_local = points.copy()
    best_local_radius = np.pi

    for step in range(2000):
        if time.time() - start > time_budget:
            break

        radii, centers, simplices = compute_facet_data(points)
        if len(radii) == 0:
            break

        max_radius = np.max(radii)

        if max_radius < best_local_radius:
            best_local_radius = max_radius
            best_local = points.copy()

        # Only move the largest facets (top 10%)
        if len(radii) > 5:
            threshold = np.percentile(radii, 90)
            mask = radii > threshold
        else:
            max_idx = np.argmax(radii)
            mask = np.zeros(len(radii), dtype=bool)
            mask[max_idx] = True

        if not np.any(mask):
            break

        updates = np.zeros_like(points)

        for m in np.where(mask)[0]:
            simplex = simplices[m]
            i_idx, j_idx, k_idx = simplex

            a = points[i_idx]
            b = points[j_idx]
            c = points[k_idx]

            center = centers[m]

            step_size = alpha * radii[m]

            updates[i_idx] += step_size * (center - a)
            updates[j_idx] += step_size * (center - b)
            updates[k_idx] += step_size * (center - c)

        points += updates
        points /= np.linalg.norm(points, axis=1, keepdims=True)

        # Decay step size
        alpha *= 0.995

        if alpha < 1e-5:
            break

    return best_local


def smooth_optimize(points, time_budget):
    """
    Smooth epigraph optimization using soft-max (log-sum-exp) surrogate.
    Minimize t + beta * log-sum-exp((r_i - t)/beta) via projected gradient on sphere.
    """
    start = time.time()
    best = points.copy()
    best_val = _covering_radius_rad(best)
    n = len(points)

    # Adaptive beta: start coarse, refine
    beta_schedule = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    lr = 0.01
    momentum = np.zeros_like(points)
    mu = 0.9  # momentum coefficient

    for beta in beta_schedule:
        if time.time() - start > time_budget:
            break

        # Reset momentum for new beta
        momentum = np.zeros_like(points)
        lr = 0.02

        for it in range(3000):
            if time.time() - start > time_budget:
                break

            # Compute facet radii
            radii, centers, simplices = compute_facet_data(points)
            if len(radii) == 0:
                break

            # Current covering radius
            r_max = np.max(radii)

            # Soft-max objective: t* + beta * log(sum(exp((r_i - t*)/beta)))
            # where t* = max(r_i)
            # Gradient w.r.t. t: 1 - sum(exp((r_i-t)/beta)) / sum(exp((r_i-t)/beta)) = 0
            # So optimal t = r_max, and objective = r_max + beta * log(M) where M = num facets
            # But for gradient computation, we use the soft-max form:
            # f = beta * logsumexp((r_i - r_max)/beta) + r_max
            # df/dx_j = (1/beta) * sum_i exp((r_i - r_max)/beta) * dr_i/dx_j

            # Compute weights for soft-max
            shifted = (radii - r_max) / beta
            # Clip for numerical stability
            shifted = np.clip(shifted, -50, 50)
            exp_vals = np.exp(shifted)
            weights = exp_vals / np.sum(exp_vals)

            # Gradient of each facet radius w.r.t. its 3 vertices
            # For a facet with circumcenter direction u and radius r:
            # r = arccos(u . a) where a is a vertex
            # dr/da = -u / sqrt(1 - (u.a)^2) projected to tangent plane
            # But since points are on sphere, we work in tangent space.
            # Simpler: use finite difference or the analytical gradient.
            # dr/dp_i for vertex i of facet m:
            # r_m = arccos(u_m . p_i)
            # dr_m/dp_i = -u_m / sin(r_m)  (in R^3, then project to tangent)
            # Tangent projection: grad_t = grad - (grad . p_i) * p_i

            grad = np.zeros_like(points)
            for m in range(len(radii)):
                w = weights[m]
                if w < 1e-10:
                    continue
                r_m = radii[m]
                if r_m < 1e-6:
                    continue
                sin_r = np.sin(r_m)
                if sin_r < 1e-8:
                    continue
                center = centers[m]
                simplex = simplices[m]
                for idx in simplex:
                    p = points[idx]
                    # dr/dp = -center / sin(r)
                    raw_grad = -center / sin_r
                    # Project to tangent plane at p
                    grad_t = raw_grad - np.dot(raw_grad, p) * p
                    grad[idx] += w * grad_t

            # Nesterov momentum
            momentum = mu * momentum + grad
            points = points - lr * momentum
            points = _normalize(points)

            # Adaptive learning rate
            new_r_max = _covering_radius_rad(points)
            if new_r_max < best_val - 1e-8:
                best_val = new_r_max
                best = points.copy()
            elif new_r_max > r_max + 0.01:
                # Step too large, reduce lr
                lr *= 0.7
                points = best.copy()
                momentum = np.zeros_like(points)

            # Check convergence
            if it > 100 and abs(new_r_max - r_max) < 1e-10:
                break

    return best

def construct():
    start_time = time.time()
    time_limit = 145.0

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
    for s in range(20):
        np.random.seed(1000 + s * 7)
        init = init_fib + np.random.randn(N, 3) * (0.1 + 0.1 * s / 20)
        init /= np.linalg.norm(init, axis=1, keepdims=True)
        initializations.append(init)

    max_restarts = 25

    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break

        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            np.random.seed(restart * 137)
            points = init_fib + np.random.randn(N, 3) * 0.3
            points /= np.linalg.norm(points, axis=1, keepdims=True)

        # Local Optimization (Lloyd-like)
        alpha = 0.05
        no_improve = 0

        for step in range(5000):
            if time.time() - start_time > time_limit:
                break

            radii, centers, simplices = compute_facet_data(points)

            if len(radii) == 0:
                break

            max_radius = np.max(radii)

            if max_radius < best_radius:
                best_radius = max_radius
                best_points = points.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 50:
                    alpha *= 0.9
                    if alpha < 1e-4:
                        break
                if no_improve > 200:
                    break

            # Identify facets to move
            threshold = np.percentile(radii, 80)
            mask = radii > threshold

            if not np.any(mask):
                max_idx = np.argmax(radii)
                mask = np.zeros(len(radii), dtype=bool)
                mask[max_idx] = True

            # Accumulate moves for vertices
            updates = np.zeros_like(points)

            for m in np.where(mask)[0]:
                simplex = simplices[m]
                i_idx, j_idx, k_idx = simplex

                a = points[i_idx]
                b = points[j_idx]
                c = points[k_idx]

                center = centers[m]

                step_size = alpha * radii[m]

                updates[i_idx] += step_size * (center - a)
                updates[j_idx] += step_size * (center - b)
                updates[k_idx] += step_size * (center - c)

            points += updates
            points /= np.linalg.norm(points, axis=1, keepdims=True)

    # Smooth optimization phase on the best configuration
    if best_points is not None and time.time() - start_time < time_limit - 10.0:
        remaining = time_limit - 10.0 - (time.time() - start_time)
        if remaining > 3.0:
            smoothed = smooth_optimize(best_points, remaining)
            sm_radius = _covering_radius_rad(smoothed)
            if sm_radius < best_radius:
                best_radius = sm_radius
                best_points = smoothed

    # Final high-precision polish
    if best_points is not None and time.time() - start_time < time_limit - 5.0:
        remaining_time = time_limit - 5.0 - (time.time() - start_time)
        if remaining_time > 2.0:
            polished = polish(best_points, remaining_time)
            radii, _, _ = compute_facet_data(polished)
            if len(radii) > 0:
                pol_radius = np.max(radii)
                if pol_radius < best_radius:
                    best_radius = pol_radius
                    best_points = polished

    if best_points is None:
        best_points = init_fib

    return best_points
