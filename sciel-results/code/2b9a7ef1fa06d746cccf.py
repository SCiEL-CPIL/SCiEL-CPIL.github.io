import numpy as np
import scipy.spatial
import time

N = 50

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii in one pass.
    Returns: hull, centers (array), radii (array), worst_idx
    """
    norms = np.linalg.norm(points, axis=1, keepdims=True)
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

def _repulsive_force(points, k_rep=5.0, d_min=0.05):
    """
    Compute repulsive force on each point from all other points.
    Force is along the great-circle direction away from neighbors,
    with magnitude k_rep / max(d, d_min)^2.
    This is a smooth approximation.
    """
    n = len(points)
    forces = np.zeros_like(points)

    # Compute pairwise angular distances
    # For efficiency with N=50, we can do this in O(N^2)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = np.arccos(np.clip(np.dot(points[i], points[j]), -1.0, 1.0))
            d_eff = max(d, d_min)
            # Direction from j to i on the sphere is (p_i - (p_i.p_j)p_j) normalized
            # This is the tangent direction
            proj = np.dot(points[i], points[j]) * points[j]
            tangent = points[i] - proj
            norm_t = np.linalg.norm(tangent)
            if norm_t > 1e-10:
                tangent = tangent / norm_t
            else:
                # Points are very close, pick an arbitrary tangent
                # This shouldn't happen often with d_min
                continue
            force_mag = k_rep / (d_eff * d_eff)
            forces[i] += force_mag * tangent

    return forces

def _pressure_towards_worst(points, hull, centers, radii, k_press=1.0, top_k=5):
    """
    Compute a force pushing points away from the circumcenters of the worst facets.
    The idea: if a facet has a large radius, we want to move its vertices
    in a way that reduces that radius. A heuristic is to push vertices away
    from the circumcenter? No, that increases distance.
    Actually, to reduce the circumradius of a triangle on the sphere,
    we should make the triangle "more equilateral" or smaller.
    A better heuristic: the gradient of the radius with respect to vertex p
    is roughly in the direction of the circumcenter C.
    So moving p towards C reduces the distance from p to C?
    Wait, R = angle(p, C). If we move p towards C, R decreases.
    So the force should be TOWARDS C for the vertices of the worst facets.
    """
    n = len(points)
    forces = np.zeros_like(points)

    if hull is None or radii is None:
        return forces

    # Sort facets by radius descending
    sorted_indices = np.argsort(radii)[::-1]
    count = 0
    for idx in sorted_indices:
        if count >= top_k:
            break
        if radii[idx] < 1e-6:
            continue
        simplex = hull.simplices[idx]
        C = centers[idx]
        r = radii[idx]

        # Weight by how bad this facet is relative to the worst
        weight = radii[idx] / np.max(radii)

        for k in range(3):
            p_idx = simplex[k]
            p = points[p_idx]
            # Direction from p to C
            proj = np.dot(p, C) * C
            tangent = C - proj
            norm_t = np.linalg.norm(tangent)
            if norm_t > 1e-10:
                tangent = tangent / norm_t
            else:
                continue
            # Force magnitude scaled by weight
            force_mag = k_press * weight
            forces[p_idx] += force_mag * tangent
        count += 1

    return forces

def _combined_forces(points, k_rep=5.0, k_press=1.0, top_k=5):
    """
    Combine repulsive and pressure forces.
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    f_rep = _repulsive_force(points, k_rep=k_rep)
    f_press = _pressure_towards_worst(points, hull, centers, radii, k_press=k_press, top_k=top_k)
    return f_rep + f_press, hull, radii

def _local_search_repulsive(points, time_budget, k_rep=5.0, k_press=1.0, top_k=5):
    """
    Local search using the combined repulsive/pressure force system.
    """
    current_points = points.copy()
    start_time = time.time()

    # Initial evaluation
    hull, centers, radii, worst_idx = _compute_facet_data(current_points)
    current_radius = np.max(radii) if worst_idx is not None else np.pi
    best_radius = current_radius
    best_points = current_points.copy()

    step_size = 0.05
    min_step = 1e-6
    decay = 0.9
    increase = 1.1

    while time.time() - start_time < time_budget:
        # Compute forces
        forces, hull, radii = _combined_forces(current_points, k_rep=k_rep, k_press=k_press, top_k=top_k)

        # Normalize forces to unit length for stable steps
        force_norm = np.linalg.norm(forces)
        if force_norm < 1e-10:
            break
        forces = forces / force_norm

        # Take a step
        new_points = current_points + step_size * forces
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)

        # Evaluate new radius
        new_hull, new_centers, new_radii, new_worst = _compute_facet_data(new_points)
        new_radius = np.max(new_radii) if new_worst is not None else np.pi

        if new_radius < current_radius:
            current_points = new_points
            current_radius = new_radius
            step_size = min(step_size * increase, 0.2)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
        else:
            step_size *= decay
            if step_size < min_step:
                break

    return best_points, best_radius

def _local_search_gradient(points, time_budget, beta=100.0):
    """
    Local search using gradient descent on soft-max radius (from seed).
    """
    current_points = points.copy()
    start_time = time.time()

    # Initial evaluation
    hull, centers, radii, worst_idx = _compute_facet_data(current_points)
    current_radius = np.max(radii) if worst_idx is not None else np.pi
    best_radius = current_radius
    best_points = current_points.copy()

    step_size = 0.03
    min_step = 1e-6
    decay = 0.95

    while time.time() - start_time < time_budget:
        # Compute soft-max radius and gradients
        hull, centers, radii, worst_idx = _compute_facet_data(current_points)
        if hull is None or worst_idx is None:
            break

        max_r = np.max(radii)
        exp_terms = np.exp(beta * (radii - max_r))
        log_sum = np.log(np.sum(exp_terms))
        soft_radius = max_r + log_sum / beta

        weights = exp_terms / np.sum(exp_terms)

        grads = np.zeros_like(current_points)
        for i in range(len(hull.simplices)):
            if radii[i] < 1e-10 or weights[i] < 1e-12:
                continue
            w = weights[i]
            C = centers[i]
            r = radii[i]
            sin_r = np.sin(r)
            if sin_r < 1e-10:
                continue
            grad_contrib = -C / sin_r
            simplex = hull.simplices[i]
            for k in range(3):
                p_idx = simplex[k]
                grads[p_idx] += w * grad_contrib

        if soft_radius < best_radius - 1e-12:
            best_radius = soft_radius
            best_points = current_points.copy()

        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-12:
            break

        grads = grads / grad_norm

        new_points = current_points + step_size * grads
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)

        # Check improvement using hard radius for stability
        new_hull, new_centers, new_radii, new_worst = _compute_facet_data(new_points)
        new_radius = np.max(new_radii) if new_worst is not None else np.pi

        if new_radius < current_radius:
            current_points = new_points
            current_radius = new_radius
            step_size = min(step_size * 1.1, 0.1)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
        else:
            step_size *= decay
            if step_size < min_step:
                break

    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_points = None
    best_radius = np.inf

    seeds = []

    # 1. Fibonacci Sphere
    j = np.arange(N) + 0.5
    z = 1.0 - 2.0 * j / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    seeds.append(fib)

    # 2. Random starts
    for seed_val in [42, 1, 2, 3, 7, 13]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(pts)

    n_seeds = len(seeds)

    # Phase 1: Repulsive force local search on all seeds
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 10:
            break

        if i < n_seeds - 1:
            time_budget = min(8.0, remaining / (n_seeds - i))
        else:
            time_budget = min(20.0, remaining)

        if time_budget < 2.0:
            break

        pts, rad = _local_search_repulsive(seed_pts, time_budget=time_budget, k_rep=5.0, k_press=1.0, top_k=5)
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()

    # Phase 2: Gradient descent polish on the best result
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 5:
            budget = min(30.0, remaining - 2)
            if budget > 2:
                pts, rad = _local_search_gradient(best_points, time_budget=budget, beta=100.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()

    # Phase 3: Final repulsive polish with different parameters
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 5:
            budget = min(20.0, remaining - 2)
            if budget > 2:
                pts, rad = _local_search_repulsive(best_points, time_budget=budget, k_rep=10.0, k_press=0.5, top_k=10)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()

    if best_points is None:
        best_points = seeds[0]

    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms

    return best_points
