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
        Objective for L-BFGS-B: sum of squared distances from each point to its nearest facet circumcenter.
        This is a smooth surrogate that encourages even distribution.
        """
        if time.time() > deadline - 10:
            return 1e10

        pts = params.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms

        facets = compute_facet_circumcenters(pts)
        if not facets:
            return 1e10

        # For each point, find the minimum distance to any facet circumcenter
        # This is a bit expensive, but helps establish good global structure
        total = 0.0
        for i in range(N):
            p = pts[i]
            min_d2 = np.inf
            for radius, center, simplex in facets:
                # Only consider facets that contain point i or are nearby
                d2 = np.sum((p - center) ** 2)
                if d2 < min_d2:
                    min_d2 = d2
            total += min_d2

        return total

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

            # Find the facet corresponding to worst_center
            try:
                hull = ConvexHull(pts)
            except:
                break

            target_indices = None
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
                if np.dot(c, worst_center) > 0.9999:
                    target_indices = simplex
                    break

            if target_indices is None:
                break

            # Move vertices towards worst_center in tangent space
            for idx in target_indices:
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
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005]
    for step in step_sizes:
        if time.time() > deadline - 15:
            break

        current = optimize_worst_facet(current, step, iterations=80)
        radius, _ = compute_covering_radius(current)

        if radius < best_radius:
            best_radius = radius
            best = current.copy()
        else:
            current = best.copy()

    # Phase 4: Random restarts with small perturbations
    if time.time() < deadline - 20:
        for restart in range(15):
            if time.time() > deadline - 20:
                break

            # Create a perturbed version of the best solution
            perturbed = best.copy()
            noise_scale = 0.03 / (restart + 1)
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])

            # Optimize the perturbed solution
            for step in [0.02, 0.01]:
                if time.time() > deadline - 20:
                    break
                perturbed = optimize_worst_facet(perturbed, step, iterations=40)

            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 5: Final high-precision polish via adaptive step size in tangent space
    if time.time() < deadline - 5:
        final = best.copy()
        # Use adaptive step size that decreases as we converge
        step = 0.005
        prev_radius = compute_covering_radius(final)[0]
        for _ in range(200):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(final)

            # If no improvement, reduce step size
            if radius >= prev_radius:
                step *= 0.5
                if step < 1e-10:
                    break
            else:
                prev_radius = radius

            try:
                hull = ConvexHull(final)
            except:
                break

            target_indices = None
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
                if np.dot(c, worst_center) > 0.9999:
                    target_indices = simplex
                    break

            if target_indices is None:
                break

            for idx in target_indices:
                v = final[idx]
                dir_vec = worst_center - v
                # Project onto tangent space: remove radial component
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                dir_vec_norm = np.linalg.norm(dir_vec)
                if dir_vec_norm > 1e-10:
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
