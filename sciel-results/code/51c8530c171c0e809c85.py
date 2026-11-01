"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

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

    def sum_sq_min_dist_objective(params):
        """
        Objective for L-BFGS-B: smooth surrogate based on soft-max of facet circumradii.
        This directly targets minimizing the covering radius.
        """
        if time.time() > deadline - 10:
            return 1e10

        pts = params.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts = pts / norms

        facets = compute_facet_circumcenters(pts)
        if not facets:
            return 1e10

        radii = np.array([r for r, _, _ in facets])
        if len(radii) == 0:
            return 1e10

        # Soft-max approximation: tau * log(sum(exp(radii/tau)))
        # This is a smooth upper bound on max(radii)
        tau = 0.02
        scaled = radii / tau
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        return tau * log_sum_exp

    def max_angular_distance_objective(params):
        """
        Objective for final polish: the maximum angular distance (covering radius).
        """
        if time.time() > deadline - 5:
            return 1e10

        pts = params.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms

        radius, _ = compute_covering_radius(pts)
        return radius

    def optimize_worst_facet(pts, step_size, iterations=50):
        """
        Iteratively move the vertices of the worst facet towards the worst center.
        Also considers nearby facets to avoid local minima.
        Uses adaptive step size and tangent space projection.
        """
        pts = pts.copy()
        prev_radius = np.pi
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(pts)

            # If radius hasn't improved, reduce step size
            if radius >= prev_radius:
                step_size *= 0.5
                if step_size < 1e-8:
                    break
            else:
                prev_radius = radius

            # Find facets corresponding to worst_center and nearby centers
            try:
                hull = ConvexHull(pts)
            except:
                break

            # Collect all facets and their circumcenters
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

            # Find facets that are close to the worst center (within threshold)
            target_indices_set = set()
            for c, simplex in facet_data:
                if np.dot(c, worst_center) > 0.999:
                    for idx in simplex:
                        target_indices_set.add(idx)

            if not target_indices_set:
                break

            # Move vertices towards worst_center in tangent space
            for idx in target_indices_set:
                v = pts[idx]
                dir_vec = worst_center - v
                # Project onto tangent space: remove radial component
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

    # Phase 2: L-BFGS-B optimization on the smooth surrogate objective
    if time.time() < deadline - 30:
        try:
            x0 = current.flatten()
            result = minimize(
                sum_sq_min_dist_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success or result.fun < sum_sq_min_dist_objective(x0):
                lbfgs_pts = project_to_sphere(result.x.reshape(N, 3))
                lbfgs_radius, _ = compute_covering_radius(lbfgs_pts)
                if lbfgs_radius < best_radius:
                    best_radius = lbfgs_radius
                    best = lbfgs_pts.copy()
                    current = lbfgs_pts.copy()
        except Exception:
            pass

    # Phase 3: Worst-facet optimization with adaptive step size
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
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

    # Phase 4: Random restarts with small perturbations
    if time.time() < deadline - 20:
        np.random.seed(42)  # For reproducibility
        for restart in range(40):
            if time.time() > deadline - 20:
                break

            # Create a perturbed version of the best solution
            perturbed = best.copy()
            noise_scale = 0.02 / (restart + 1)
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])

            # Optimize the perturbed solution
            for step in [0.012, 0.006, 0.003, 0.001]:
                if time.time() > deadline - 20:
                    break
                perturbed = optimize_worst_facet(perturbed, step, iterations=50)

            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 5: Final high-precision polish via adaptive step size in tangent space
    if time.time() < deadline - 5:
        final = best.copy()
        # Use adaptive step size that decreases as we converge
        step = 0.001
        prev_radius = compute_covering_radius(final)[0]
        stall_count = 0
        for it in range(800):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(final)

            # If no improvement, reduce step size
            if radius >= prev_radius - 1e-14:
                step *= 0.5
                stall_count += 1
                if step < 1e-14 or stall_count > 30:
                    break
            else:
                prev_radius = radius
                stall_count = 0

            try:
                hull = ConvexHull(final)
            except:
                break

            # Find all facets close to worst_center (use tighter threshold)
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
                if np.dot(c, worst_center) > 0.9998:
                    for idx in simplex:
                        target_indices_set.add(idx)

            if not target_indices_set:
                break

            for idx in target_indices_set:
                v = final[idx]
                dir_vec = worst_center - v
                # Project onto tangent space: remove radial component
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

    # Ensure all points are non-zero and finite
    norms = np.linalg.norm(best, axis=1)
    valid = (norms > 1e-10) & np.all(np.isfinite(best), axis=1)
    if not np.all(valid):
        # Fallback to Fibonacci if something went wrong
        best = fibonacci_sphere(N)

    return best
# EVOLVE-BLOCK-END
