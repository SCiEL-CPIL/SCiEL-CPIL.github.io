import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_facet_data(points):
    """
    Compute all facet circumcenters and circumradii.
    Returns: (max_radius, worst_facet_idx, circumcenters, radii)
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, -1, None, None

    simplices = hull.simplices
    v1 = points[simplices[:, 0]]
    v2 = points[simplices[:, 1]]
    v3 = points[simplices[:, 2]]

    n = np.cross(v2 - v1, v3 - v1)
    norms = np.linalg.norm(n, axis=1)
    valid = norms > 1e-10
    if not np.any(valid):
        return np.pi, -1, None, None

    idx_valid = np.where(valid)[0]
    n_valid = n[idx_valid]
    norms_valid = norms[idx_valid]
    v1_valid = v1[idx_valid]
    v2_valid = v2[idx_valid]
    v3_valid = v3[idx_valid]

    c_circ = n_valid / norms_valid[:, None]
    centroids = (v1_valid + v2_valid + v3_valid) / 3.0
    dots = np.sum(c_circ * centroids, axis=1)
    flip = dots < 0
    c_circ[flip] = -c_circ[flip]

    cos_theta = np.sum(c_circ * v1_valid, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)

    max_idx = np.argmax(radii)
    max_radius = radii[max_idx]
    worst_facet = simplices[idx_valid[max_idx]]

    return max_radius, worst_facet, c_circ, radii

def _smooth_covering_radius(pts, grid, beta1=30.0, beta2=30.0):
    """
    Smooth surrogate for the covering radius using soft-max.
    Returns (value, grad) where grad is (N, 3).
    """
    dots = grid @ pts.T  # (M, N)
    shifted = beta1 * dots
    max_shifted = np.max(shifted, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted - max_shifted)
    weights = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    g = np.sum(weights * dots, axis=1)
    g = np.clip(g, -1.0 + 1e-10, 1.0 - 1e-10)
    d = np.arccos(g)

    shifted2 = beta2 * d
    max_shifted2 = np.max(shifted2)
    exp_shifted2 = np.exp(shifted2 - max_shifted2)
    y_weights = exp_shifted2 / np.sum(exp_shifted2)

    value = np.sum(y_weights * d)

    sqrt_term = np.sqrt(np.maximum(1.0 - g**2, 1e-12))
    inv_sqrt = 1.0 / sqrt_term

    grad = np.zeros((N, 3))
    for i in range(N):
        weighted = y_weights * weights[:, i] * inv_sqrt
        grad[i] = -np.sum(weighted[:, None] * grid, axis=0)

    # Project to tangent space
    for i in range(N):
        grad[i] = grad[i] - np.dot(grad[i], pts[i]) * pts[i]

    return value, grad

def _repulsive_energy(pts, k=1.0):
    """
    Repulsive energy to prevent clustering.
    Returns (energy, grad).
    """
    # Pairwise dot products
    dots = pts @ pts.T
    np.fill_diagonal(dots, 0.0)

    # Use 1/(1-dots) as repulsive potential
    # grad w.r.t. pts[i] = k * sum_j (1/(1-dots_ij)^2) * (dots_ij * pts[i] - pts[j])
    rep = k / (1.0 - dots)**2
    np.fill_diagonal(rep, 0.0)

    energy = np.sum(rep)

    # Gradient: for each i, grad_i = k * sum_j rep_ij * (dots_ij * pts[i] - pts[j])
    grad = np.zeros((N, 3))
    for i in range(N):
        # sum_j rep[i,j] * (dots[i,j] * pts[i] - pts[j])
        term1 = np.sum(rep[i] * dots[i]) * pts[i]
        term2 = rep[i] @ pts
        grad[i] = k * (term1 - term2)

    # Project to tangent space
    for i in range(N):
        grad[i] = grad[i] - np.dot(grad[i], pts[i]) * pts[i]

    return energy, grad

def _combined_objective(pts, grid, beta1, beta2, rep_k):
    """
    Combined objective: smooth covering radius + repulsive energy.
    Returns (value, grad).
    """
    val1, grad1 = _smooth_covering_radius(pts, grid, beta1, beta2)
    val2, grad2 = _repulsive_energy(pts, rep_k)
    return val1 + val2, grad1 + grad2

def _local_optimize_smooth(pts, grid, beta1=30.0, beta2=30.0, rep_k=0.5, deadline=None):
    """
    Optimize the smooth surrogate using L-BFGS-B on the sphere.
    """
    def objective_and_grad(x):
        pts = x.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        pts = pts / norms
        val, grad = _combined_objective(pts, grid, beta1, beta2, rep_k)
        # Project gradient to tangent space
        for i in range(N):
            grad[i] = grad[i] - np.dot(grad[i], pts[i]) * pts[i]
        return val, grad.flatten()

    x0 = pts.flatten()
    result = minimize(objective_and_grad, x0, method='L-BFGS-B',
                      jac=True, options={'maxiter': 200, 'maxfun': 5000,
                                         'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-8})
    pts_opt = result.x.reshape(N, 3)
    pts_opt = pts_opt / np.linalg.norm(pts_opt, axis=1, keepdims=True)
    return pts_opt

def _local_refine_facet(pts, max_iters=3000, step_size=0.05, decay=0.9995, deadline=None):
    """Local refinement: move worst facet vertices toward circumcenter."""
    current_radius, worst_facet, _, _ = _compute_facet_data(pts)
    best_local = pts.copy()
    best_local_radius = current_radius

    alpha = step_size
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        current_radius, worst_facet, worst_center, _ = _compute_facet_data(pts)

        if current_radius < best_local_radius:
            best_local_radius = current_radius
            best_local = pts.copy()

        if worst_facet is None or len(worst_facet) != 3:
            break

        v1_idx, v2_idx, v3_idx = worst_facet
        v1, v2, v3 = pts[v1_idx], pts[v2_idx], pts[v3_idx]

        p1_new = v1 + alpha * (worst_center - v1)
        p2_new = v2 + alpha * (worst_center - v2)
        p3_new = v3 + alpha * (worst_center - v3)

        n1 = np.linalg.norm(p1_new)
        n2 = np.linalg.norm(p2_new)
        n3 = np.linalg.norm(p3_new)

        if n1 > 1e-10 and n2 > 1e-10 and n3 > 1e-10:
            pts[v1_idx] = p1_new / n1
            pts[v2_idx] = p2_new / n2
            pts[v3_idx] = p3_new / n3

        alpha *= decay
        if alpha < 1e-8:
            break

    return best_local, best_local_radius

def _gradient_descent(pts, grid, max_iters=500, step_size=0.01, decay=0.999, deadline=None, beta1=30.0, beta2=30.0, rep_k=0.5):
    """Gradient descent on the smooth covering radius + repulsion."""
    best = pts.copy()
    best_radius, _, _, _ = _compute_facet_data(best)

    alpha = step_size
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        val, grad = _combined_objective(pts, grid, beta1, beta2, rep_k)

        # Normalize gradient
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm = np.maximum(grad_norm, 1e-12)
        grad = grad / grad_norm

        new_pts = pts + alpha * grad
        new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)

        new_radius, _, _, _ = _compute_facet_data(new_pts)

        if new_radius < best_radius:
            best_radius = new_radius
            best = new_pts.copy()
            pts = new_pts
        else:
            alpha *= 0.9
            if alpha < 1e-6:
                break

        alpha *= decay

    return best, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Initial configuration
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_radius, _, _, _ = _compute_facet_data(best_points)

    np.random.seed(42)

    # Create a grid for smooth objective
    M_grid = 1500
    i_grid = np.arange(M_grid) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / M_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])

    # Phase 1: Rapid local refinement from Fibonacci
    if time.time() < deadline - 100:
        best_points, best_radius = _local_refine_facet(points, max_iters=3000, step_size=0.05, decay=0.9995, deadline=deadline)

    # Phase 2: Gradient descent from best
    if time.time() < deadline - 80:
        gd_pts, gd_radius = _gradient_descent(best_points, grid, max_iters=300, step_size=0.01, decay=0.999, deadline=deadline)
        if gd_radius < best_radius:
            best_radius = gd_radius
            best_points = gd_pts.copy()

    # Phase 3: Multiple restarts with mixed strategies
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 40:
            break

        # Generate initial points
        if restart == 0:
            pts = best_points.copy()
        elif restart == 1:
            pts = _fibonacci_sphere(N)
            noise = np.random.randn(N, 3) * 0.1
            pts = pts + noise
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif restart == 2:
            # Random points
            pts = np.random.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif restart == 3:
            # Mix with best solution
            pts = np.random.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            mix = 0.5
            pts = (1 - mix) * pts + mix * best_points
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif restart == 4:
            # Perturbed best
            pts = best_points + np.random.randn(N, 3) * 0.05
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif restart == 5:
            # Fibonacci with larger noise
            pts = _fibonacci_sphere(N)
            noise = np.random.randn(N, 3) * 0.2
            pts = pts + noise
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif restart == 6:
            # Two-point mirror
            pts = _fibonacci_sphere(N)
            # Randomly flip some points
            flip_mask = np.random.rand(N) < 0.3
            pts[flip_mask] = -pts[flip_mask]
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            # Random with repulsion initialization
            pts = np.random.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)

        current_radius, _, _, _ = _compute_facet_data(pts)

        # Simulated annealing with gradient steps
        temperature = 0.03
        cooling = 0.997
        min_temperature = 1e-5
        max_sa_iters = 1500
        stuck_counter = 0
        best_sa_radius = current_radius
        best_sa_pts = pts.copy()

        for it in range(max_sa_iters):
            if time.time() > deadline - 20:
                break

            # Choose a random point to move
            idx = np.random.randint(N)

            current_radius, worst_facet, worst_center, all_radii = _compute_facet_data(pts)

            if worst_facet is not None and len(worst_facet) == 3:
                v1_idx, v2_idx, v3_idx = worst_facet
                if idx in [v1_idx, v2_idx, v3_idx]:
                    target = worst_center
                    move = target - pts[idx]
                    move_norm = np.linalg.norm(move)
                    if move_norm > 1e-10:
                        move = move / move_norm * temperature
                else:
                    move = np.random.randn(3) * temperature
            else:
                move = np.random.randn(3) * temperature

            new_pt = pts[idx] + move
            new_pt_norm = np.linalg.norm(new_pt)
            if new_pt_norm < 1e-10:
                continue
            new_pt = new_pt / new_pt_norm

            old_pt = pts[idx].copy()
            pts[idx] = new_pt

            new_radius, _, _, _ = _compute_facet_data(pts)

            delta = new_radius - current_radius
            if delta < 0 or np.random.rand() < np.exp(-delta / max(temperature, 1e-10)):
                current_radius = new_radius
                if new_radius < best_sa_radius:
                    best_sa_radius = new_radius
                    best_sa_pts = pts.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                pts[idx] = old_pt
                stuck_counter += 1

            temperature *= cooling
            if temperature < min_temperature:
                break

            # If stuck, do a gradient step
            if stuck_counter > 100 and it % 30 == 0:
                val, grad = _combined_objective(pts, grid, 30.0, 30.0, 0.5)
                grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
                grad_norm = np.maximum(grad_norm, 1e-12)
                grad = grad / grad_norm
                pts = pts + 0.005 * grad
                pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
                current_radius, _, _, _ = _compute_facet_data(pts)
                if current_radius < best_sa_radius:
                    best_sa_radius = current_radius
                    best_sa_pts = pts.copy()
                    stuck_counter = 0

        # Update global best
        if best_sa_radius < best_radius:
            best_radius = best_sa_radius
            best_points = best_sa_pts.copy()

        # Local refine the best from this restart
        if time.time() < deadline - 15:
            refined_pts, refined_radius = _local_refine_facet(best_sa_pts, max_iters=500, step_size=0.02, decay=0.9995, deadline=deadline)
            if refined_radius < best_radius:
                best_radius = refined_radius
                best_points = refined_pts.copy()

    # Phase 4: Final gradient polish
    if time.time() < deadline - 10:
        gd_pts, gd_radius = _gradient_descent(best_points, grid, max_iters=200, step_size=0.005, decay=0.9995, deadline=deadline)
        if gd_radius < best_radius:
            best_radius = gd_radius
            best_points = gd_pts.copy()

    # Phase 5: Final local polish
    if time.time() < deadline - 5:
        final_pts, final_radius = _local_refine_facet(best_points, max_iters=2000, step_size=0.005, decay=0.9995, deadline=deadline)
        if final_radius < best_radius:
            best_radius = final_radius
            best_points = final_pts.copy()

    # Ensure final points are normalized and valid
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    return best_points
