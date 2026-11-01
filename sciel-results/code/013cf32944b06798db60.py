"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit (35s buffer)

    def fibonacci_sphere(n, offset=0.5, phase_shift=0.0):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + phase_shift)
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def generate_grid_points(n_grid=10000):
        """Generate a dense quasi-uniform grid on the sphere for SLSQP constraints."""
        # Use Fibonacci spiral for quasi-uniform distribution
        i = np.arange(n_grid)
        z = 1.0 - 2.0 * (i + 0.5) / n_grid
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def compute_hull_data(pts):
        """
        Compute all facet circumcenters and their angular radii.
        Returns: (facets, circumcenters, radii) or (None, None, None) on failure.
        """
        try:
            hull = ConvexHull(pts)
            facets = hull.simplices
        except Exception:
            return None, None, None

        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None

        a = pts[facets[:, 0]]
        b = pts[facets[:, 1]]
        c = pts[facets[:, 2]]

        v1 = a - b
        v2 = b - c
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1)

        valid = norm_n > 1e-12
        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)

        if np.any(valid):
            cc = n[valid] / norm_n[valid, np.newaxis]
            dots = np.sum(cc * a[valid], axis=1)
            cc[dots < 0] *= -1
            circumcenters[valid] = cc

            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)

        return facets, circumcenters, radii

    def compute_covering_radius(pts):
        """
        Compute the covering radius in radians and the worst-case circumcenter.
        """
        facets, circumcenters, radii = compute_hull_data(pts)
        if radii is None:
            return np.pi, np.array([0.0, 0.0, 1.0])

        max_idx = np.argmax(radii)
        max_radius = radii[max_idx]
        worst_center = circumcenters[max_idx]

        return max_radius, worst_center

    def project_to_sphere(pts):
        """Project points onto the unit sphere."""
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return pts / norms

    def max_angular_distance_to_grid(pts, grid):
        """
        Compute the maximum angular distance from any grid point to its nearest site.
        Returns (max_dist, worst_grid_idx).
        """
        # pts: (N, 3), grid: (M, 3)
        # Compute all pairwise dot products: (N, M)
        dots = grid @ pts.T  # (M, N)
        # For each grid point, find the max dot (min angular distance)
        max_dots = np.max(dots, axis=1)  # (M,)
        # Clip for numerical safety
        max_dots = np.clip(max_dots, -1.0, 1.0)
        angular_dists = np.arccos(max_dots)  # (M,)
        max_idx = np.argmax(angular_dists)
        return angular_dists[max_idx], max_idx

    def slsqp_objective_and_constraints(x, grid, t0):
        """
        SLSQP: minimize t subject to max angular distance <= t.
        x = [x1..x3, y1..y3, ..., t] where first 3*N are site coords, last is t.
        """
        n_sites = N
        site_coords = x[:3 * n_sites].reshape(n_sites, 3)
        t = x[3 * n_sites]

        # Normalize sites
        norms = np.linalg.norm(site_coords, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        sites = site_coords / norms

        # Compute max angular distance to grid
        max_dist, _ = max_angular_distance_to_grid(sites, grid)

        # Objective: minimize t
        obj = t

        # Constraint: max_dist - t <= 0
        constraint_val = max_dist - t

        return obj, constraint_val

    def slsqp_minimize(x0, grid, t0, maxiter=200):
        """
        Run SLSQP to minimize t subject to max angular distance <= t.
        """
        n_sites = N
        x0 = np.concatenate([x0.flatten(), [t0]])

        def objective(x):
            if time.time() > deadline - 10:
                return 1e10
            obj, _ = slsqp_objective_and_constraints(x, grid, t0)
            return obj

        def constraint_func(x):
            if time.time() > deadline - 10:
                return -1e10
            _, c = slsqp_objective_and_constraints(x, grid, t0)
            return c

        constraints = [{'type': 'ineq', 'fun': constraint_func}]

        # Bounds: sites on sphere (each coord in [-1, 1]), t in [0, pi]
        bounds = [(-1.0, 1.0)] * (3 * n_sites) + [(0.0, np.pi)]

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'disp': False}
        )

        return result

    def optimize_worst_facet(pts, step_size, iterations=60):
        """
        Iteratively move the vertices of the worst facet towards the worst center.
        Uses adaptive step size and tangent space projection.
        """
        pts = pts.copy()
        prev_radius = np.pi
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(pts)

            if radius >= prev_radius:
                step_size *= 0.5
                if step_size < 1e-8:
                    break
            else:
                prev_radius = radius

            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break

            # Find facets that are close to the worst center
            target_indices_set = set()
            for i in range(len(facets)):
                if np.dot(circumcenters[i], worst_center) > 0.999:
                    for idx in facets[i]:
                        target_indices_set.add(idx)

            if not target_indices_set:
                break

            for idx in target_indices_set:
                v = pts[idx]
                dir_vec = worst_center - v
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    pts[idx] = v + step_size * dir_vec
                    pts[idx] /= np.linalg.norm(pts[idx])

        return pts

    # Phase 1: Initial Fibonacci spiral
    current = fibonacci_sphere(N)
    best = current.copy()
    best_radius, _ = compute_covering_radius(best)

    # Generate grid for SLSQP
    grid = generate_grid_points(8000)

    # Phase 2: SLSQP optimization
    if time.time() < deadline - 50:
        try:
            t0 = best_radius + 0.05  # Initial t slightly above current radius
            result = slsqp_minimize(current, grid, t0, maxiter=150)
            if result.success or result.fun < t0:
                slsqp_pts = project_to_sphere(result.x[:3 * N].reshape(N, 3))
                slsqp_radius, _ = compute_covering_radius(slsqp_pts)
                if slsqp_radius < best_radius:
                    best_radius = slsqp_radius
                    best = slsqp_pts.copy()
                    current = slsqp_pts.copy()
        except Exception:
            pass

    # Phase 3: Worst-facet optimization with adaptive step size
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005]
    for step in step_sizes:
        if time.time() > deadline - 25:
            break

        current = optimize_worst_facet(current, step, iterations=50)
        radius, _ = compute_covering_radius(current)

        if radius < best_radius:
            best_radius = radius
            best = current.copy()
        else:
            current = best.copy()

    # Phase 4: Second SLSQP pass from improved starting point
    if time.time() < deadline - 30:
        try:
            t0 = best_radius + 0.02
            result = slsqp_minimize(best, grid, t0, maxiter=100)
            if result.success or result.fun < t0:
                slsqp_pts = project_to_sphere(result.x[:3 * N].reshape(N, 3))
                slsqp_radius, _ = compute_covering_radius(slsqp_pts)
                if slsqp_radius < best_radius:
                    best_radius = slsqp_radius
                    best = slsqp_pts.copy()
                    current = slsqp_pts.copy()
        except Exception:
            pass

    # Phase 5: Random restarts with small perturbations + hull refinement
    if time.time() < deadline - 20:
        np.random.seed(42)
        for restart in range(15):
            if time.time() > deadline - 20:
                break

            perturbed = best.copy()
            noise_scale = 0.015 / (restart + 1)
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])

            for step in [0.01, 0.005, 0.002]:
                if time.time() > deadline - 20:
                    break
                perturbed = optimize_worst_facet(perturbed, step, iterations=30)

            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 6: Final high-precision polish via adaptive step size in tangent space
    if time.time() < deadline - 5:
        final = best.copy()
        step = 0.002
        prev_radius = compute_covering_radius(final)[0]
        stall_count = 0
        for it in range(400):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(final)

            if radius >= prev_radius - 1e-12:
                step *= 0.5
                stall_count += 1
                if step < 1e-12 or stall_count > 15:
                    break
            else:
                prev_radius = radius
                stall_count = 0

            facets, circumcenters, radii = compute_hull_data(final)
            if facets is None:
                break

            target_indices_set = set()
            for i in range(len(facets)):
                if np.dot(circumcenters[i], worst_center) > 0.9995:
                    for idx in facets[i]:
                        target_indices_set.add(idx)

            if not target_indices_set:
                break

            for idx in target_indices_set:
                v = final[idx]
                dir_vec = worst_center - v
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                dir_vec_norm = np.linalg.norm(dir_vec)
                if dir_vec_norm > 1e-12:
                    dir_vec /= dir_vec_norm
                    final[idx] = v + step * dir_vec
                    final[idx] /= np.linalg.norm(final[idx])

        radius, _ = compute_covering_radius(final)
        if radius < best_radius:
            best_radius = radius
            best = final

    # Final normalization and validation
    best = project_to_sphere(best)

    norms = np.linalg.norm(best, axis=1)
    valid = (norms > 1e-10) & np.all(np.isfinite(best), axis=1)
    if not np.all(valid):
        best = fibonacci_sphere(N)

    return best
# EVOLVE-BLOCK-END
