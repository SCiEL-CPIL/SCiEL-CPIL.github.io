import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _compute_facet_data(points):
    """
    Compute convex hull facets and their circumradii.
    Returns: (radii, circumcenters, simplices, valid_mask)
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None, None

    simplices = hull.simplices  # (F, 3)
    v1 = points[simplices[:, 0]]
    v2 = points[simplices[:, 1]]
    v3 = points[simplices[:, 2]]

    n_vec = np.cross(v2 - v1, v3 - v1)
    norms = np.linalg.norm(n_vec, axis=1)
    valid = norms > 1e-10
    if not np.any(valid):
        return None, None, None, None

    idx_valid = np.where(valid)[0]
    n_valid = n_vec[idx_valid]
    norms_valid = norms[idx_valid]
    v1_valid = v1[idx_valid]
    v2_valid = v2[idx_valid]
    v3_valid = v3[idx_valid]

    c_circ = n_valid / norms_valid[:, None]

    # Ensure circumcenter points outward (same side as centroid)
    centroids = (v1_valid + v2_valid + v3_valid) / 3.0
    dots = np.sum(c_circ * centroids, axis=1)
    flip = dots < 0
    c_circ[flip] = -c_circ[flip]

    cos_theta = np.sum(c_circ * v1_valid, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)

    return radii, c_circ, simplices[idx_valid], valid

def _covering_radius(points):
    """Return the covering radius in radians."""
    radii, _, _, _ = _compute_facet_data(points)
    if radii is None:
        return np.pi
    return np.max(radii)

def _smooth_covering_radius(points, beta=50.0):
    """
    Smooth surrogate for the max circumradius using soft-max.
    Returns (smooth_value, all_radii, circumcenters, simplices)
    """
    radii, c_circ, simplices, valid = _compute_facet_data(points)
    if radii is None:
        return np.pi, None, None, None

    # Soft-max approximation with log-sum-exp for numerical stability
    max_r = np.max(radii)
    shifted = radii - max_r
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    smooth_val = max_r + np.log(np.sum(weights)) / beta

    return smooth_val, radii, c_circ, simplices

def _riemannian_gradient_step(points, beta=50.0, learning_rate=0.01, top_k=5):
    """
    Compute a Riemannian gradient step to reduce the smooth covering radius.
    Focuses gradient on top-k worst facets for precision.
    Returns new points after the step.
    """
    radii, c_circ, simplices, valid = _compute_facet_data(points)
    if radii is None:
        return points.copy()

    n_facets = len(radii)
    
    # Use soft-max weights but focus on top-k facets
    max_r = np.max(radii)
    shifted = radii - max_r
    weights = np.exp(beta * shifted)
    total_w = np.sum(weights)
    weights /= total_w

    # Also compute hard top-k weights for focused gradient
    top_k_idx = np.argsort(radii)[-top_k:]
    top_k_weights = np.zeros(n_facets)
    top_k_weights[top_k_idx] = 1.0 / top_k

    # Blend: 70% top-k focused, 30% soft-max
    blended_weights = 0.7 * top_k_weights + 0.3 * weights

    # Gradient of circumradius w.r.t. vertices on the sphere:
    # For a facet with circumcenter c and vertices v1, v2, v3,
    # The spherical gradient of arccos(c·vi) at vi is:
    # -(c - vi*(c·vi)) / sin(theta)
    # This is the projection of -(c - vi*cos(theta)) onto the tangent space

    grad = np.zeros_like(points)
    sin_theta = np.sqrt(np.maximum(1.0 - np.cos(radii)**2, 1e-12))
    cos_theta = np.cos(radii)

    for i in range(n_facets):
        w = blended_weights[i]
        if w < 1e-6:
            continue
        s = sin_theta[i]
        if s < 1e-10:
            continue
        c = c_circ[i]
        ct = cos_theta[i]
        
        # Gradient contribution for each vertex in this facet
        # d(arccos(c·vi))/d(vi) projected to tangent space
        for k in range(3):
            vi_idx = simplices[i, k]
            vi = points[vi_idx]
            # Tangent projection of -(c - vi*ct)
            g = -(c - vi * ct)
            # Project to tangent space: g - (g·vi)*vi
            g_proj = g - np.dot(g, vi) * vi
            grad[vi_idx] += w * g_proj

    # Take step in tangent space (already projected)
    new_points = points - learning_rate * grad
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    new_points /= norms

    return new_points

def _line_search(points, lr, beta=50.0, max_backtracks=15, top_k=5):
    """
    Perform backtracking line search to find a good learning rate.
    Returns (new_points, used_lr, improved)
    """
    current_cr = _covering_radius(points)

    for bt in range(max_backtracks):
        new_pts = _riemannian_gradient_step(points, beta, lr, top_k)
        new_cr = _covering_radius(new_pts)

        if new_cr < current_cr - 1e-12:
            return new_pts, lr, True
        lr *= 0.5
        if lr < 1e-8:
            break

    return points, lr, False

def _local_optimize(points, beta=50.0, max_iters=300, lr_init=0.05, deadline=None, top_k=5):
    """
    Local optimization using Riemannian gradient descent with adaptive parameters.
    """
    best_pts = points.copy()
    best_cr = _covering_radius(best_pts)
    lr = lr_init
    no_improve_count = 0

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        # Adaptive beta: increase as we converge
        effective_beta = beta * (1.0 + 0.5 * it / max_iters)
        # Adaptive top_k: start broad, focus narrow
        effective_top_k = max(3, top_k - it // 50)

        # Try to improve
        new_pts, lr, improved = _line_search(points, lr, effective_beta, top_k=effective_top_k)

        if improved:
            no_improve_count = 0
            points = new_pts
            cr = _covering_radius(points)
            if cr < best_cr:
                best_cr = cr
                best_pts = points.copy()
        else:
            no_improve_count += 1
            lr *= 0.7
            if lr < 1e-10:
                break
            
            # If stuck for a while, try perturbation
            if no_improve_count >= 15:
                n_perturb = np.random.randint(1, 3)
                for _ in range(n_perturb):
                    idx = np.random.randint(N)
                    noise = np.random.randn(3) * 0.015
                    p_new = points[idx] + noise
                    n_new = np.linalg.norm(p_new)
                    if n_new > 1e-10:
                        points[idx] = p_new / n_new
                no_improve_count = 0
                lr = max(lr, 0.02)

    return best_pts, best_cr

def _swap_optimize(points, deadline=None):
    """
    Try swapping pairs of points to see if it reduces the covering radius.
    """
    best_pts = points.copy()
    best_cr = _covering_radius(best_pts)

    if deadline is not None and time.time() > deadline:
        return best_pts, best_cr

    # Try random swaps
    for attempt in range(150):
        if deadline is not None and time.time() > deadline:
            break

        i, j = np.random.choice(N, 2, replace=False)
        new_pts = best_pts.copy()
        new_pts[i], new_pts[j] = best_pts[j], best_pts[i]

        cr = _covering_radius(new_pts)
        if cr < best_cr - 1e-10:
            best_pts = new_pts
            best_cr = cr

    return best_pts, best_cr

def _perturb_and_optimize(points, deadline=None):
    """
    Create perturbed copies of the points and optimize each.
    Returns the best result.
    """
    best_pts = points.copy()
    best_cr = _covering_radius(best_pts)

    if deadline is not None and time.time() > deadline:
        return best_pts, best_cr

    n_perturbations = 4

    for p in range(n_perturbations):
        if deadline is not None and time.time() > deadline - 15:
            break

        # Create perturbation
        perturbed = best_pts.copy()
        n_pert = np.random.randint(2, 5)
        for _ in range(n_pert):
            idx = np.random.randint(N)
            noise = np.random.randn(3) * 0.04
            p_new = perturbed[idx] + noise
            n_new = np.linalg.norm(p_new)
            if n_new > 1e-10:
                perturbed[idx] = p_new / n_new

        # Optimize the perturbed version
        opt_pts, opt_cr = _local_optimize(perturbed, beta=40.0, max_iters=80, lr_init=0.03, deadline=deadline - 10, top_k=5)
        if opt_cr < best_cr:
            best_pts = opt_pts
            best_cr = opt_cr

    return best_pts, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    np.random.seed(42)

    best_points = None
    best_radius = np.pi

    def update_best(pts):
        nonlocal best_points, best_radius
        cr = _covering_radius(pts)
        if cr < best_radius:
            best_radius = cr
            best_points = pts.copy()
        return cr

    # Phase 1: Multiple diverse initializations with local optimization
    initializations = [
        _fibonacci_sphere(N),
        _random_sphere(N, seed=1),
        _random_sphere(N, seed=2),
        _random_sphere(N, seed=3),
        _fibonacci_sphere(N) + np.random.randn(N, 3) * 0.05,
        _random_sphere(N, seed=4),
        _random_sphere(N, seed=5),
        _random_sphere(N, seed=6),
        _random_sphere(N, seed=7),
        _random_sphere(N, seed=8),
    ]

    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 40:
            break

        # Normalize
        norms = np.linalg.norm(init_pts, axis=1, keepdims=True)
        init_pts = init_pts / np.where(norms < 1e-10, 1.0, norms)

        cr = update_best(init_pts)

        # Local optimization with adaptive parameters
        opt_pts, opt_cr = _local_optimize(init_pts, beta=30.0, max_iters=250, lr_init=0.05, deadline=deadline - 35, top_k=5)
        update_best(opt_pts)

        # Second round with higher beta for finer tuning
        if time.time() < deadline - 35:
            opt_pts2, opt_cr2 = _local_optimize(opt_pts, beta=80.0, max_iters=150, lr_init=0.02, deadline=deadline - 30, top_k=4)
            update_best(opt_pts2)

    # Phase 2: Focused refinement on best solution with increasing beta
    if best_points is not None and time.time() < deadline - 20:
        pts = best_points.copy()
        for beta_val in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 10:
                break
            opt_pts, opt_cr = _local_optimize(pts, beta=beta_val, max_iters=120, lr_init=0.01, deadline=deadline - 10, top_k=3)
            update_best(opt_pts)
            pts = opt_pts

    # Phase 3: Swap optimization to escape local minima
    if best_points is not None and time.time() < deadline - 15:
        swap_pts, swap_cr = _swap_optimize(best_points, deadline=deadline - 10)
        update_best(swap_pts)

    # Phase 4: Perturbation and re-optimization
    if best_points is not None and time.time() < deadline - 10:
        perturb_pts, perturb_cr = _perturb_and_optimize(best_points, deadline=deadline - 5)
        update_best(perturb_pts)

    # Phase 5: Final high-precision polish with very narrow focus
    if best_points is not None and time.time() < deadline - 5:
        pts = best_points.copy()
        lr = 0.005
        for it in range(150):
            if time.time() > deadline - 2:
                break
            new_pts, lr, improved = _line_search(pts, lr, beta=300.0, max_backtracks=8, top_k=3)
            if improved:
                pts = new_pts
                cr = _covering_radius(pts)
                if cr < best_radius:
                    best_radius = cr
                    best_points = pts.copy()
            else:
                lr *= 0.5
                if lr < 1e-10:
                    break

    # Final validation and normalization
    if best_points is None:
        best_points = _fibonacci_sphere(N)

    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)

    # Ensure no coincident points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-6:
                axis = np.cross(best_points[i], best_points[j])
                if np.linalg.norm(axis) < 1e-10:
                    axis = np.array([1.0, 0.0, 0.0])
                axis /= np.linalg.norm(axis)
                theta = 0.01
                c, s = np.cos(theta), np.sin(theta)
                R = c * np.eye(3) + (1-c) * np.outer(axis, axis) + s * np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                best_points[j] = R @ best_points[j]
                best_points[j] /= np.linalg.norm(best_points[j])

    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)

    return best_points
