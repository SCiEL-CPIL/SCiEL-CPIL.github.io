import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 130

def _compute_covering_radius(points):
    """
    Computes the covering radius (max circumradius of hull facets) in degrees.
    Returns (radius_in_degrees, worst_facet_index, worst_circumcenter, hull, all_radii, all_centers, all_facets).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0, -1, None, None, None, None, None

    max_radius = 0.0
    worst_facet_idx = -1
    worst_center = None
    radii = []
    centers = []
    facets = []

    for i, simplex in enumerate(hull.simplices):
        u = points[simplex[0]]
        v = points[simplex[1]]
        w = points[simplex[2]]

        uv = np.dot(u, v)
        uw = np.dot(u, w)
        vw = np.dot(v, w)

        M = np.array([
            [1.0, uv, uw],
            [uv, 1.0, vw],
            [uw, vw, 1.0]
        ])

        try:
            x = np.linalg.solve(M, np.ones(3))
        except np.linalg.LinAlgError:
            continue

        xT_M_x = np.dot(x, M @ x)

        if xT_M_x <= 1e-10:
            continue

        cos_R = 1.0 / np.sqrt(xT_M_x)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        R = math.acos(cos_R)

        C = x[0] * u + x[1] * v + x[2] * w
        norm_C = np.linalg.norm(C)
        if norm_C < 1e-10:
            continue
        C = C / norm_C

        radii.append(R)
        centers.append(C)
        facets.append(simplex)

        if R > max_radius:
            max_radius = R
            worst_center = C
            worst_facet_idx = i

    if not radii:
        return 90.0, -1, None, None, None, None, None

    return math.degrees(max_radius), worst_facet_idx, worst_center, hull, \
           np.array(radii), np.array(centers), np.array(facets)

def _move_point_towards(p, c, step_size):
    """Move point p on the sphere towards c by step_size radians."""
    dot_pc = np.dot(p, c)
    t = c - dot_pc * p
    norm_t = np.linalg.norm(t)
    if norm_t < 1e-10:
        return p
    t = t / norm_t
    new_p = p + step_size * t
    new_p = new_p / np.linalg.norm(new_p)
    return new_p

def _fibonacci_spiral(n):
    """Generate a Fibonacci spiral configuration on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def _repulsion_step(points, alpha):
    """Perform one repulsion step to spread points out (vectorized)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)

    inv_dist_cubed = 1.0 / (dist_sq ** 1.5)
    np.fill_diagonal(inv_dist_cubed, 0.0)

    forces = np.sum(diff * inv_dist_cubed[:, :, np.newaxis], axis=1)

    forces_norm = np.linalg.norm(forces, axis=1, keepdims=True)
    forces = forces / np.where(forces_norm == 0, 1, forces_norm)

    new_points = points + alpha * forces
    new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
    return new_points

def _lloyd_iteration(points, max_iter=100, deadline=None):
    """
    Lloyd's algorithm: replace each point with the centroid of its Voronoi cell.
    """
    from scipy.spatial import Voronoi
    points = points.copy()

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        try:
            vor = Voronoi(points)
        except Exception:
            break

        new_points = np.zeros_like(points)

        for i in range(N):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]

            if len(region) == 0 or -1 in region:
                new_points[i] = points[i]
                continue

            verts = vor.vertices[region]
            verts = verts[np.all(np.isfinite(verts), axis=1)]

            if len(verts) == 0:
                new_points[i] = points[i]
                continue

            centroid = np.mean(verts, axis=0)
            norm = np.linalg.norm(centroid)
            if norm < 1e-10:
                new_points[i] = points[i]
            else:
                new_points[i] = centroid / norm

        diff = np.linalg.norm(new_points - points, axis=1)
        if np.max(diff) < 1e-8:
            points = new_points
            break

        points = new_points

    return points

def _gradient_ascent_on_worst_facet(points, hull, center, alpha):
    """
    Perform a gradient-ascent-like step: move the 3 vertices of the worst facet
    towards the circumcenter. Returns new points and whether it improved.
    """
    radius_before, facet_idx, _, _, _, _, _ = _compute_covering_radius(points)
    if facet_idx == -1 or center is None:
        return points, False

    simplex = hull.simplices[facet_idx]
    v0, v1, v2 = simplex

    new_points = points.copy()
    new_points[v0] = _move_point_towards(points[v0], center, alpha)
    new_points[v1] = _move_point_towards(points[v1], center, alpha)
    new_points[v2] = _move_point_towards(points[v2], center, alpha)

    radius_after, _, _, _, _, _, _ = _compute_covering_radius(new_points)

    if radius_after < radius_before:
        return new_points, True
    else:
        return points, False

def _soft_max_covering_radius(points, temperature=0.05):
    """
    Compute a smooth approximation of the covering radius using log-sum-exp.
    Returns (soft_max_in_degrees, weighted_center, hull, radii_array, centers_array, facets_array)
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0, None, None, None, None, None

    radii = []
    centers = []
    facets = []

    for i, simplex in enumerate(hull.simplices):
        u = points[simplex[0]]
        v = points[simplex[1]]
        w = points[simplex[2]]

        uv = np.dot(u, v)
        uw = np.dot(u, w)
        vw = np.dot(v, w)

        M = np.array([
            [1.0, uv, uw],
            [uv, 1.0, vw],
            [uw, vw, 1.0]
        ])

        try:
            x = np.linalg.solve(M, np.ones(3))
        except np.linalg.LinAlgError:
            continue

        xT_M_x = np.dot(x, M @ x)

        if xT_M_x <= 1e-10:
            continue

        cos_R = 1.0 / np.sqrt(xT_M_x)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        R = math.acos(cos_R)

        C = x[0] * u + x[1] * v + x[2] * w
        norm_C = np.linalg.norm(C)
        if norm_C < 1e-10:
            continue
        C = C / norm_C

        radii.append(R)
        centers.append(C)
        facets.append(simplex)

    if not radii:
        return 90.0, None, None, None, None, None

    radii = np.array(radii)
    centers_arr = np.array(centers)
    facets_arr = np.array(facets)

    max_r = np.max(radii)
    if max_r < 1e-10:
        return 0.0, None, None, radii, centers_arr, facets_arr

    exp_terms = np.exp((radii - max_r) / temperature)
    sum_exp = np.sum(exp_terms)
    soft_max = max_r + temperature * np.log(sum_exp)

    weights = exp_terms / sum_exp
    weighted_center = np.sum(weights[:, np.newaxis] * centers_arr, axis=0)
    weighted_center_norm = np.linalg.norm(weighted_center)
    if weighted_center_norm < 1e-10:
        weighted_center = centers_arr[np.argmax(radii)]
    else:
        weighted_center = weighted_center / weighted_center_norm

    return math.degrees(soft_max), weighted_center, hull, radii, centers_arr, facets_arr

def _multi_facet_gradient_search(points, deadline, temperature=0.05, max_iter=400):
    """
    Use soft-max covering radius as a smooth surrogate for gradient-based search.
    Move points in the direction that reduces the soft-max radius.
    """
    points = points.copy()
    best_radius, _, _, _, _, _, _ = _compute_covering_radius(points)
    best_points = points.copy()

    for it in range(max_iter):
        if time.time() > deadline:
            break

        soft_rad, center, hull, radii, centers_arr, facets_arr = _soft_max_covering_radius(points, temperature)

        if center is None or hull is None:
            break

        actual_radius, _, _, _, _, _, _ = _compute_covering_radius(points)
        if actual_radius < best_radius:
            best_radius = actual_radius
            best_points = points.copy()

        forces = np.zeros_like(points)

        if radii is not None and len(radii) > 0:
            max_r = np.max(radii)
            exp_terms = np.exp((radii - max_r) / temperature)
            weights = exp_terms / np.sum(exp_terms)

            for i in range(len(facets_arr)):
                simplex = facets_arr[i]
                w = weights[i]
                c = centers_arr[i]
                for v_idx in simplex:
                    p = points[v_idx]
                    dot_pc = np.dot(p, c)
                    t = c - dot_pc * p
                    norm_t = np.linalg.norm(t)
                    if norm_t > 1e-10:
                        t = t / norm_t
                        forces[v_idx] += w * t

        forces_norm = np.linalg.norm(forces, axis=1, keepdims=True)
        forces = forces / np.where(forces_norm == 0, 1, forces_norm)

        step_size = 0.04 * (1.0 + 0.5 * math.cos(0.1 * it))

        new_points = points + step_size * forces
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)

        new_radius, _, _, _, _, _, _ = _compute_covering_radius(new_points)

        if new_radius < actual_radius:
            points = new_points
        else:
            for mult in [0.5, 0.25, 0.1]:
                test_points = points + step_size * mult * forces
                test_points = test_points / np.linalg.norm(test_points, axis=1, keepdims=True)
                test_radius, _, _, _, _, _, _ = _compute_covering_radius(test_points)
                if test_radius < actual_radius:
                    points = test_points
                    break
            else:
                if it % 50 == 49:
                    noise = np.random.randn(N, 3) * 0.03
                    points = points + noise
                    points = points / np.linalg.norm(points, axis=1, keepdims=True)
                else:
                    temperature *= 1.05

    return best_points, best_radius

def _spectral_optimization(points, deadline, max_iter=200):
    """
    Spectral method: use the eigenvector of the Laplacian to spread points.
    This is a heuristic that works well for sphere covering.
    """
    points = points.copy()

    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Build a graph where edges connect nearby points
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1e10)

        # Connect each point to its k nearest neighbors
        k = 10
        knn_idx = np.argsort(dist_sq, axis=1)[:, :k]

        # Build adjacency matrix
        adj = np.zeros((N, N))
        for i in range(N):
            for j in knn_idx[i]:
                adj[i, j] = 1.0

        # Laplacian
        deg = np.sum(adj, axis=1)
        L = np.diag(deg) - adj

        # Solve for the second eigenvector (Fiedler vector)
        try:
            from scipy.linalg import eigh
            eigenvalues, eigenvectors = eigh(L)
            # The second eigenvector (index 1) gives the best cut
            fiedler = eigenvectors[:, 1]
        except Exception:
            break

        # Use the Fiedler vector to adjust points
        # Points with high Fiedler value move away from center, etc.
        fiedler_mean = np.mean(fiedler)
        fiedler_std = np.std(fiedler)
        if fiedler_std < 1e-10:
            break

        # Normalize
        fiedler_norm = (fiedler - fiedler_mean) / fiedler_std

        # Adjust step size
        step = 0.02 * (1.0 + 0.3 * math.cos(0.05 * it))

        new_points = points.copy()
        for i in range(N):
            # Move point in the direction that increases its Fiedler value
            # This helps spread points apart
            direction = points[i] * fiedler_norm[i]
            new_points[i] = points[i] + step * direction
            new_points[i] = new_points[i] / np.linalg.norm(new_points[i])

        radius, _, _, _, _, _, _ = _compute_covering_radius(new_points)
        if radius < _compute_covering_radius(points)[0]:
            points = new_points

    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    np.random.seed(42)

    best_points = None
    best_radius = 90.0

    # Phase 1: Global optimization via repulsion from multiple starts
    for seed in range(6):
        if time.time() > deadline - 60:
            break

        np.random.seed(seed * 7919 + 42)

        # Start with Fibonacci spiral plus jitter
        points = _fibonacci_spiral(N).copy()
        noise = np.random.randn(N, 3) * (0.02 + 0.01 * seed)
        points = points + noise
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        # Global repulsion to spread points
        alpha = 0.03
        for step in range(120):
            if time.time() > deadline - 60:
                break

            points = _repulsion_step(points, alpha)
            alpha *= 0.995

            if step % 15 == 0 or step == 119:
                radius, _, _, _, _, _, _ = _compute_covering_radius(points)
                if radius < best_radius:
                    best_radius = radius
                    best_points = points.copy()

    # Phase 2: Lloyd's algorithm to improve distribution
    if best_points is not None:
        points = best_points.copy()
    else:
        points = _fibonacci_spiral(N)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        best_radius, _, _, _, _, _, _ = _compute_covering_radius(points)
        best_points = points.copy()

    if time.time() < deadline - 45:
        points = _lloyd_iteration(points, max_iter=60, deadline=deadline - 40)
        radius, _, _, _, _, _, _ = _compute_covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()

    # Phase 3: Multi-facet gradient search using smooth surrogate
    if best_points is not None and time.time() < deadline - 25:
        points, soft_radius = _multi_facet_gradient_search(best_points, deadline=deadline - 25, temperature=0.05, max_iter=250)
        if soft_radius < best_radius:
            best_radius = soft_radius
            best_points = points.copy()

    # Phase 4: Local search from the best point found
    if best_points is not None:
        points = best_points.copy()
    else:
        points = _fibonacci_spiral(N)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        best_radius, _, _, _, _, _, _ = _compute_covering_radius(points)
        best_points = points.copy()

    alpha = 0.08
    stagnation_counter = 0
    max_stagnation = 40

    while time.time() < deadline - 15:
        radius, facet_idx, center, hull, _, _, _ = _compute_covering_radius(points)

        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if center is None or facet_idx == -1:
            break

        simplex = hull.simplices[facet_idx]
        v0, v1, v2 = simplex

        improved = False

        # Try moving all 3 vertices of the worst facet towards its circumcenter
        new_points, imp = _gradient_ascent_on_worst_facet(points, hull, center, alpha)
        if imp:
            points = new_points
            improved = True
        else:
            # Try moving vertices individually
            for v_idx in [v0, v1, v2]:
                test_points = points.copy()
                test_points[v_idx] = _move_point_towards(points[v_idx], center, alpha)
                test_radius, _, _, _, _, _, _ = _compute_covering_radius(test_points)
                if test_radius < radius:
                    points = test_points
                    improved = True
                    break

        if not improved:
            alpha *= 0.8
            if alpha < 0.003:
                # Perturb the worst facet's vertices to escape local minimum
                noise = np.random.randn(3, 3) * 0.05
                new_points = points.copy()
                new_points[v0] = points[v0] + noise[0]
                new_points[v1] = points[v1] + noise[1]
                new_points[v2] = points[v2] + noise[2]

                for idx in [v0, v1, v2]:
                    new_points[idx] = new_points[idx] / np.linalg.norm(new_points[idx])

                pert_radius, _, _, _, _, _, _ = _compute_covering_radius(new_points)

                if pert_radius < best_radius * 1.02:
                    points = new_points
                    alpha = 0.06
                    stagnation_counter = 0
                else:
                    if stagnation_counter > max_stagnation:
                        break
                    else:
                        alpha = 0.02

    # Phase 5: Another Lloyd pass for final improvement
    if time.time() < deadline - 10 and best_points is not None:
        points = best_points.copy()
        points = _lloyd_iteration(points, max_iter=30, deadline=deadline - 8)
        radius, _, _, _, _, _, _ = _compute_covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()

    # Phase 6: High-precision polish pass with adaptive step sizes
    if time.time() < deadline - 5 and best_points is not None:
        points = best_points.copy()
        polish_alpha = 0.015
        for polish_iter in range(80):
            if time.time() > deadline - 5:
                break

            radius, facet_idx, center, hull, _, _, _ = _compute_covering_radius(points)
            if center is None or facet_idx == -1:
                break

            simplex = hull.simplices[facet_idx]
            v0, v1, v2 = simplex

            improved = False
            for step_mult in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]:
                step = polish_alpha * step_mult
                new_points = points.copy()
                new_points[v0] = _move_point_towards(points[v0], center, step)
                new_points[v1] = _move_point_towards(points[v1], center, step)
                new_points[v2] = _move_point_towards(points[v2], center, step)

                new_radius, _, _, _, _, _, _ = _compute_covering_radius(new_points)
                if new_radius < radius:
                    points = new_points
                    improved = True
                    break

            if not improved:
                for v_idx in [v0, v1, v2]:
                    for step_mult in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]:
                        step = polish_alpha * step_mult
                        test_points = points.copy()
                        test_points[v_idx] = _move_point_towards(points[v_idx], center, step)
                        test_radius, _, _, _, _, _, _ = _compute_covering_radius(test_points)
                        if test_radius < radius:
                            points = test_points
                            improved = True
                            break
                    if improved:
                        break

            if not improved:
                polish_alpha *= 0.65
                if polish_alpha < 0.0002:
                    break
            else:
                if radius < best_radius:
                    best_radius = radius
                    best_points = points.copy()

    # Phase 7: Final fine-tuning with very small steps
    if time.time() < deadline - 2 and best_points is not None:
        points = best_points.copy()
        for final_iter in range(40):
            if time.time() > deadline - 2:
                break

            radius, facet_idx, center, hull, _, _, _ = _compute_covering_radius(points)
            if center is None or facet_idx == -1:
                break

            simplex = hull.simplices[facet_idx]
            v0, v1, v2 = simplex

            improved = False
            for step in [0.004, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
                new_points = points.copy()
                new_points[v0] = _move_point_towards(points[v0], center, step)
                new_points[v1] = _move_point_towards(points[v1], center, step)
                new_points[v2] = _move_point_towards(points[v2], center, step)

                new_radius, _, _, _, _, _, _ = _compute_covering_radius(new_points)
                if new_radius < radius:
                    points = new_points
                    improved = True
                    break

            if not improved:
                for v_idx in [v0, v1, v2]:
                    for step in [0.002, 0.001, 0.0005, 0.0002]:
                        test_points = points.copy()
                        test_points[v_idx] = _move_point_towards(points[v_idx], center, step)
                        test_radius, _, _, _, _, _, _ = _compute_covering_radius(test_points)
                        if test_radius < radius:
                            points = test_points
                            improved = True
                            break
                    if improved:
                        break

            if not improved:
                break
            else:
                if radius < best_radius:
                    best_radius = radius
                    best_points = points.copy()

    return best_points
