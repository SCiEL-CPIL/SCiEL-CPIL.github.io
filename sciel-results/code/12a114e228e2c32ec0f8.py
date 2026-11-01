import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 50

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii in one pass.
    Returns: hull, centers (array), radii (array), worst_idx
    """
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    points = points / norms

    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return None, None, None, None

    n_facets = len(hull.simplices)
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)

    for i, simplex in enumerate(hull.simplices):
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]

        ab = np.dot(a, b)
        ac = np.dot(a, c)
        bc = np.dot(b, c)

        Gram = np.array([
            [1.0, ab, ac],
            [ab, 1.0, bc],
            [ac, bc, 1.0]
        ])

        try:
            m = np.linalg.solve(Gram, np.ones(3))
            c_dir = m[0]*a + m[1]*b + m[2]*c
            norm_c_dir = np.linalg.norm(c_dir)
            if norm_c_dir < 1e-10:
                continue
            centers[i] = c_dir / norm_c_dir
            radii[i] = np.arccos(np.clip(np.dot(centers[i], a), -1.0, 1.0))
        except np.linalg.LinAlgError:
            continue

    if n_facets > 0 and np.any(radii > 0):
        worst_idx = np.argmax(radii)
    else:
        worst_idx = None

    return hull, centers, radii, worst_idx

def _compute_soft_max_radius_and_grads(points, beta=100.0):
    """
    Compute soft-max of facet radii and gradients.
    Soft-max: R_soft = (1/beta) * log(sum(exp(beta * r_i)))
    This is smooth and differentiable.
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)

    if hull is None or worst_idx is None:
        return np.pi, np.zeros_like(points)

    # Soft-max radius
    max_r = np.max(radii)
    # Use log-sum-exp for numerical stability
    exp_terms = np.exp(beta * (radii - max_r))
    log_sum = np.log(np.sum(exp_terms))
    soft_radius = max_r + np.log(np.sum(exp_terms)) / beta

    # Weights for each facet (softmax of radii)
    weights = exp_terms / np.sum(exp_terms)

    # Gradients: d(soft_radius)/d(p_j) = sum_i w_i * d(r_i)/d(p_j)
    grads = np.zeros_like(points)

    for i in range(len(hull.simplices)):
        if radii[i] < 1e-10 or weights[i] < 1e-12:
            continue

        w = weights[i]
        C = centers[i]
        r = radii[i]
        sin_r = np.sin(r)
        if sin_r < 1e-10:
            continue

        # Gradient contribution from this facet
        grad_contrib = -C / sin_r

        simplex = hull.simplices[i]
        for k in range(3):
            p_idx = simplex[k]
            grads[p_idx] += w * grad_contrib

    return soft_radius, grads

def _local_search(points, time_budget, use_soft_max=True, beta=100.0):
    """
    Local search using gradient descent with adaptive step size.
    """
    current_points = points.copy()

    if use_soft_max:
        current_radius, _ = _compute_soft_max_radius_and_grads(current_points, beta)
    else:
        _, centers, radii, worst_idx = _compute_facet_data(current_points)
        current_radius = np.max(radii) if worst_idx is not None else np.pi

    best_radius = current_radius
    best_points = current_points.copy()

    start_time = time.time()
    step_size = 0.03
    min_step = 1e-6
    decay = 0.95

    while time.time() - start_time < time_budget:
        if use_soft_max:
            radius, grads = _compute_soft_max_radius_and_grads(current_points, beta)
        else:
            hull, centers, radii, worst_idx = _compute_facet_data(current_points)
            if hull is None or worst_idx is None:
                break
            radius = radii[worst_idx]
            grads = np.zeros_like(current_points)
            C = centers[worst_idx]
            r = radii[worst_idx]
            if r > 1e-10:
                sin_r = np.sin(r)
                if sin_r > 1e-10:
                    grad_contrib = -C / sin_r
                    simplex = hull.simplices[worst_idx]
                    for k in range(3):
                        p_idx = simplex[k]
                        grads[p_idx] += grad_contrib

        if radius < best_radius - 1e-12:
            best_radius = radius
            best_points = current_points.copy()

        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-12:
            break

        grads = grads / grad_norm

        # Try step
        new_points = current_points + step_size * grads
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)

        # Check if improvement
        if use_soft_max:
            new_radius, _ = _compute_soft_max_radius_and_grads(new_points, beta)
        else:
            _, _, new_radii, new_worst = _compute_facet_data(new_points)
            new_radius = np.max(new_radii) if new_worst is not None else np.pi

        if new_radius < radius:
            current_points = new_points
            step_size = min(step_size * 1.1, 0.1)
        else:
            step_size *= decay
            if step_size < min_step:
                break

    return best_points, best_radius

def _fibonacci_sphere(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts

def _repulsive_force_energy(flat_pts, alpha=2.0, gamma=1.0):
    """
    Energy function combining repulsive forces between points and
    pressure away from facet circumcenters.
    E = sum_{i<j} 1/d(i,j)^alpha + gamma * sum_i min_k d(i, C_k)^2
    where C_k are facet circumcenters.
    """
    pts = flat_pts.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    pts = pts / norms

    # Compute facet circumcenters
    hull, centers, radii, worst_idx = _compute_facet_data(pts)
    if hull is None:
        return 1e10

    # Repulsive term: sum over pairs of 1/d^alpha
    repulsive = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-10:
                repulsive += 1e10
            else:
                repulsive += 1.0 / (d ** alpha)

    # Pressure term: for each point, min squared distance to any circumcenter
    pressure = 0.0
    if len(centers) > 0:
        for i in range(N):
            min_sq = np.inf
            for c in centers:
                sq = np.dot(pts[i] - c, pts[i] - c)
                if sq < min_sq:
                    min_sq = sq
            pressure += min_sq

    return repulsive + gamma * pressure

def _repulsive_force_gradient(flat_pts, alpha=2.0, gamma=1.0, num_steps=1e-4):
    """
    Numerical gradient of the repulsive force energy.
    """
    grad = np.zeros_like(flat_pts)
    for k in range(len(flat_pts)):
        orig = flat_pts[k]
        flat_pts[k] = orig + num_steps
        f_plus = _repulsive_force_energy(flat_pts, alpha, gamma)
        flat_pts[k] = orig - num_steps
        f_minus = _repulsive_force_energy(flat_pts, alpha, gamma)
        flat_pts[k] = orig
        grad[k] = (f_plus - f_minus) / (2.0 * num_steps)
    return grad

def _optimize_with_repulsive_forces(initial_pts, time_budget, alpha=2.0, gamma=1.0):
    """
    Optimize using L-BFGS-B on the repulsive force energy.
    """
    start_time = time.time()
    flat_initial = initial_pts.flatten()

    try:
        result = scipy.optimize.minimize(
            _repulsive_force_energy,
            flat_initial,
            args=(alpha, gamma),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 5000}
        )
        if result.success or result.fun < 1e10:
            optimized = result.x.reshape(N, 3)
            norms = np.linalg.norm(optimized, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            optimized = optimized / norms
            return optimized
    except Exception:
        pass

    return initial_pts

def _optimize_worst_facet(pts, step_size, iterations=50, deadline=None):
    """
    Iteratively move the vertices of the worst facet towards the worst center.
    """
    pts = pts.copy()
    for _ in range(iterations):
        if deadline is not None and time.time() > deadline - 5:
            break

        hull, centers, radii, worst_idx = _compute_facet_data(pts)
        if hull is None or worst_idx is None:
            break

        worst_center = centers[worst_idx]
        simplex = hull.simplices[worst_idx]

        # Move vertices towards worst_center
        for idx in simplex:
            v = pts[idx]
            dir_vec = worst_center - v
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir > 1e-10:
                dir_vec /= norm_dir
                pts[idx] = v + step_size * dir_vec
                pts[idx] /= np.linalg.norm(pts[idx])

    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_points = None
    best_radius = np.inf

    # Seeds
    seeds = []

    # 1. Fibonacci Sphere with multiple offsets
    for offset in [0.5, 0.3, 0.7]:
        seeds.append(_fibonacci_sphere(N, offset))

    # 2. Random starts with different seeds
    for seed_val in [42, 1, 2, 3, 7, 13]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(pts)

    # Process seeds with adaptive time allocation
    n_seeds = len(seeds)

    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed

        if remaining < 15:
            break

        # Allocate time based on seed type
        if i < 3:  # Fibonacci seeds get more time
            time_budget = min(20.0, remaining / (n_seeds - i))
        else:
            time_budget = min(10.0, remaining / (n_seeds - i))

        if time_budget < 5.0:
            break

        current = seed_pts.copy()

        # Phase 1: Repulsive force optimization (new structural approach)
        if time.time() < deadline - 20:
            repulsive_budget = min(10.0, time_budget * 0.4)
            if repulsive_budget > 3:
                current = _optimize_with_repulsive_forces(current, repulsive_budget, alpha=2.0, gamma=1.0)

        # Phase 2: Soft-max gradient descent
        if time.time() < deadline - 15:
            softmax_budget = min(15.0, time_budget * 0.4)
            if softmax_budget > 3:
                current, _ = _local_search(current, softmax_budget, use_soft_max=True, beta=100.0)

        # Phase 3: Worst-facet refinement
        if time.time() < deadline - 10:
            for step in [0.05, 0.02, 0.01]:
                if time.time() > deadline - 10:
                    break
                current = _optimize_worst_facet(current, step, iterations=30, deadline=deadline)

        # Evaluate
        hull, centers, radii, worst_idx = _compute_facet_data(current)
        if hull is not None and worst_idx is not None:
            radius = radii[worst_idx]
            if radius < best_radius:
                best_radius = radius
                best_points = current.copy()

    # Phase 4: Random restarts with perturbations of best solution
    if best_points is not None and time.time() < deadline - 20:
        for restart in range(10):
            if time.time() > deadline - 15:
                break

            perturbed = best_points.copy()
            noise_scale = 0.05 / (restart + 1)
            rng = np.random.default_rng(100 + restart)
            for i in range(N):
                rand_vec = rng.normal(size=3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])

            # Quick optimization
            if time.time() < deadline - 15:
                current = perturbed.copy()
                current, _ = _local_search(current, time_budget=5.0, use_soft_max=True, beta=100.0)
                current = _optimize_worst_facet(current, 0.01, iterations=20, deadline=deadline)

                hull, centers, radii, worst_idx = _compute_facet_data(current)
                if hull is not None and worst_idx is not None:
                    radius = radii[worst_idx]
                    if radius < best_radius:
                        best_radius = radius
                        best_points = current.copy()

    # Phase 5: Final high-precision polish
    if best_points is not None and time.time() < deadline - 5:
        remaining = deadline - time.time()
        if remaining > 5:
            budget = min(20.0, remaining - 2)
            if budget > 2:
                pts, rad = _local_search(best_points, time_budget=budget, use_soft_max=True, beta=200.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()

    if best_points is None:
        best_points = seeds[0]

    # Ensure output is normalized
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    best_points = best_points / norms

    return best_points
