"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def compute_facet_circumcenters(pts):
        """
        Compute all facet circumcenters and their angular radii.
        Returns: list of (radius_radians, circumcenter, simplex_indices)
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return []

        results = []
        for simplex in hull.simplices:
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]

            edge1 = v2 - v1
            edge2 = v3 - v1
            n = np.cross(edge1, edge2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                continue

            c = n / norm_n
            if np.dot(c, v1) < 0:
                c = -c

            d = np.linalg.norm(c - v1)
            radius = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
            results.append((radius, c, simplex))

        return results

    def compute_covering_radius(pts):
        """
        Compute the covering radius in radians and the worst-case circumcenter.
        """
        facets = compute_facet_circumcenters(pts)
        if not facets:
            return np.pi, np.array([0.0, 0.0, 1.0])

        max_radius = 0.0
        worst_center = None
        for radius, center, _ in facets:
            if radius > max_radius:
                max_radius = radius
                worst_center = center

        return max_radius, worst_center

    def soft_max_grid_objective(params, tau=0.02):
        """
        Smooth surrogate: soft-max of angular distances from a grid of points to their
        nearest configuration point. Directly targets covering radius.
        """
        if time.time() > deadline - 10:
            return 1e10

        pts = params.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts = pts / norms

        # Use a fine grid of evaluation points
        n_grid = 400
        grid_pts = fibonacci_sphere(n_grid)

        # Compute angular distances from each grid point to all configuration points
        dots = grid_pts @ pts.T  # shape (n_grid, N)
        cos_theta = np.clip(dots, -1.0, 1.0)
        theta = np.arccos(cos_theta)  # angular distances (n_grid, N)

        # For each grid point, find the minimum distance to any configuration point
        min_dist = np.min(theta, axis=1)  # shape (n_grid,)

        # Soft-max over the grid points
        scaled = min_dist / tau
        max_scaled = np.max(scaled)
        if max_scaled > 500:
            return float(max_scaled * tau)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        return tau * log_sum_exp

    def soft_max_grid_gradient(params, tau=0.02):
        """
        Numerical gradient for the soft-max grid objective.
        Uses forward differences for speed.
        """
        if time.time() > deadline - 10:
            return np.zeros_like(params)

        pts = params.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts = pts / norms

        grad = np.zeros_like(pts)
        eps = 1e-5
        base_val = soft_max_grid_objective(params, tau)

        for i in range(N):
            for j in range(3):
                pts_plus = pts.copy()
                pts_plus[i, j] += eps
                # Only renormalize the changed point
                n_plus = np.linalg.norm(pts_plus[i])
                if n_plus > 1e-10:
                    pts_plus[i] /= n_plus

                f_plus = soft_max_grid_objective(pts_plus.flatten(), tau)

                if np.isfinite(f_plus) and np.isfinite(base_val):
                    grad[i, j] = (f_plus - base_val) / eps

        return grad.flatten()

    def sum_sq_min_dist_objective(params):
        """
        Objective for L-BFGS-B: smooth surrogate based on soft-max of facet circumradii.
        This directly targets minimizing the covering radius.
        """
        if time.time() > deadline - 10:
            return 1e10

        pts = params.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts = pts / norms

        facets = compute_facet_circumcenters(pts)
        if not facets:
            return 1e10

        radii = np.array([r for r, _, _ in facets])
        if len(radii) == 0:
            return 1e10

        tau = 0.015
        scaled = radii / tau
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        return tau * log_sum_exp

    def optimize_worst_facet(pts, step_size, iterations=50):
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

            try:
                hull = ConvexHull(pts)
            except:
                break

            facet_data = []
            for simplex in hull.simplices:
                v1 = pts[simplex[0]]
                v2 = pts[simplex[1]]
                v3 = pts[simplex[2]]
                edge1 = v2 - v1
                edge2 = v3 - v1
                n = np.cross(edge1, edge2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-12:
                    continue
                c = n / norm_n
                if np.dot(c, v1) < 0:
                    c = -c
                facet_data.append((c, simplex))

            if not facet_data:
                break

            target_indices_set = set()
            for c, simplex in facet_data:
                if np.dot(c, worst_center) > 0.999:
                    for idx in simplex:
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

    def project_to_sphere(pts):
        """Project points onto the unit sphere."""
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return pts / norms

    # Phase 1: Initial Fibonacci spiral
    current = fibonacci_sphere(N)
    best = current.copy()
    best_radius, _ = compute_covering_radius(best)

    # Phase 2: Soft-max grid objective optimization for global structure
    if time.time() < deadline - 40:
        try:
            x0 = current.flatten()
            result_sm = minimize(
                soft_max_grid_objective,
                x0,
                jac=soft_max_grid_gradient,
                args=(0.03,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            if result_sm.success or result_sm.fun < soft_max_grid_objective(x0, 0.03):
                sm_pts = project_to_sphere(result_sm.x.reshape(N, 3))
                sm_radius, _ = compute_covering_radius(sm_pts)
                if sm_radius < best_radius:
                    best_radius = sm_radius
                    best = sm_pts.copy()
                    current = sm_pts.copy()
        except Exception:
            pass

    # Phase 3: L-BFGS-B on soft-max facet radii for refinement (with tighter tau)
    if time.time() < deadline - 25:
        try:
            x0 = current.flatten()
            for schedule_idx in range(3):
                if time.time() > deadline - 25:
                    break
                tau_val = 0.015 / (schedule_idx + 1)
                ftol = 1e-11 / (schedule_idx + 1)
                gtol = 1e-8 / (schedule_idx + 1)
                result = minimize(
                    sum_sq_min_dist_objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 2500, 'ftol': ftol, 'gtol': gtol}
                )
                if result.success or result.fun < sum_sq_min_dist_objective(x0):
                    lbfgs_pts = project_to_sphere(result.x.reshape(N, 3))
                    lbfgs_radius, _ = compute_covering_radius(lbfgs_pts)
                    if lbfgs_radius < best_radius:
                        best_radius = lbfgs_radius
                        best = lbfgs_pts.copy()
                        current = lbfgs_pts.copy()
                        x0 = best.flatten()
        except Exception:
            pass

    # Phase 4: Worst-facet optimization with adaptive step size
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    for step in step_sizes:
        if time.time() > deadline - 15:
            break
        current = optimize_worst_facet(current, step, iterations=100)
        radius, _ = compute_covering_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()
        else:
            current = best.copy()

    # Phase 5: Random restarts with small perturbations + soft-max refinement
    if time.time() < deadline - 20:
        np.random.seed(42)
        for restart in range(40):
            if time.time() > deadline - 20:
                break
            perturbed = best.copy()
            noise_scale = 0.012 / (restart + 1)
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])
            # First do worst-facet optimization
            for step in [0.01, 0.005, 0.002]:
                if time.time() > deadline - 20:
                    break
                perturbed = optimize_worst_facet(perturbed, step, iterations=30)
            # Then refine with soft-max L-BFGS-B
            if time.time() < deadline - 22:
                try:
                    x0 = perturbed.flatten()
                    result_ref = minimize(
                        soft_max_grid_objective,
                        x0,
                        jac=soft_max_grid_gradient,
                        args=(0.02,),
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'maxfun': 800, 'ftol': 1e-11, 'gtol': 1e-8}
                    )
                    if result_ref.success or result_ref.fun < soft_max_grid_objective(x0, 0.02):
                        ref_pts = project_to_sphere(result_ref.x.reshape(N, 3))
                        ref_radius, _ = compute_covering_radius(ref_pts)
                        if ref_radius < compute_covering_radius(perturbed)[0]:
                            perturbed = ref_pts
                except Exception:
                    pass
            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 6: Final high-precision polish with soft-max refinement
    if time.time() < deadline - 5:
        final = best.copy()
        # First, do a final L-BFGS-B pass with very tight parameters
        if time.time() < deadline - 8:
            try:
                x0 = final.flatten()
                result_final = minimize(
                    soft_max_grid_objective,
                    x0,
                    jac=soft_max_grid_gradient,
                    args=(0.01,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 1500, 'ftol': 1e-13, 'gtol': 1e-10}
                )
                if result_final.success or result_final.fun < soft_max_grid_objective(x0, 0.01):
                    fin_pts = project_to_sphere(result_final.x.reshape(N, 3))
                    fin_radius, _ = compute_covering_radius(fin_pts)
                    if fin_radius < compute_covering_radius(final)[0]:
                        final = fin_pts
            except Exception:
                pass

        # Then do worst-facet polish
        step = 0.0005
        prev_radius = compute_covering_radius(final)[0]
        stall_count = 0
        for it in range(500):
            if time.time() > deadline - 5:
                break
            radius, worst_center = compute_covering_radius(final)
            if radius >= prev_radius - 1e-16:
                step *= 0.7
                stall_count += 1
                if step < 1e-16 or stall_count > 40:
                    break
            else:
                prev_radius = radius
                stall_count = 0

            try:
                hull = ConvexHull(final)
            except:
                break

            target_indices_set = set()
            for simplex in hull.simplices:
                v1 = final[simplex[0]]
                v2 = final[simplex[1]]
                v3 = final[simplex[2]]
                edge1 = v2 - v1
                edge2 = v3 - v1
                n = np.cross(edge1, edge2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-12:
                    continue
                c = n / norm_n
                if np.dot(c, v1) < 0:
                    c = -c
                if np.dot(c, worst_center) > 0.99999:
                    for idx in simplex:
                        target_indices_set.add(idx)

            if not target_indices_set:
                break

            for idx in target_indices_set:
                v = final[idx]
                dir_vec = worst_center - v
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                dir_vec_norm = np.linalg.norm(dir_vec)
                if dir_vec_norm > 1e-13:
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
