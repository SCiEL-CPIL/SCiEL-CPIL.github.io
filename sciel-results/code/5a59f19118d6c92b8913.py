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
            return [(np.pi, np.array([0.0, 0.0, 1.0]), None)]

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

        if not results:
            return [(np.pi, np.array([0.0, 0.0, 1.0]), None)]

        return results

    def compute_covering_radius(pts):
        """
        Compute the covering radius using the convex hull facet circumradius method.
        Returns the covering radius in radians and the worst-case point (circumcenter of worst facet).
        """
        facets = compute_facet_circumcenters(pts)
        max_radius = 0.0
        worst_center = None
        for radius, c, _ in facets:
            if radius > max_radius:
                max_radius = radius
                worst_center = c
        if worst_center is None:
            return np.pi, np.array([0.0, 0.0, 1.0])
        return max_radius, worst_center

    def soft_max_radius_objective(flat_pts, beta=50.0):
        """
        Smooth surrogate for the max facet circumradius using log-sum-exp.
        This is differentiable and avoids the non-smoothness of the max operator.
        """
        pts = flat_pts.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms

        facets = compute_facet_circumcenters(pts)
        if not facets or len(facets) < 2:
            return np.pi

        radii = np.array([r for r, _, _ in facets])
        # Log-sum-exp for numerical stability
        max_r = np.max(radii)
        if max_r > 0:
            log_sum = max_r + np.log(np.sum(np.exp((radii - max_r) * beta)) / len(radii)) / beta
        else:
            log_sum = 0.0
        return log_sum

    def repulsive_energy_objective(flat_pts, beta=50.0):
        """
        Combined objective: soft-max covering radius + repulsive energy term.
        The repulsive term uses inverse-square angular distance to spread points,
        while the soft-max term focuses on reducing the worst facet.
        """
        pts = flat_pts.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms

        # Soft-max covering radius
        facets = compute_facet_circumcenters(pts)
        if not facets or len(facets) < 2:
            return np.pi

        radii = np.array([r for r, _, _ in facets])
        max_r = np.max(radii)
        if max_r > 0:
            log_sum = max_r + np.log(np.sum(np.exp((radii - max_r) * beta)) / len(radii)) / beta
        else:
            log_sum = 0.0

        # Repulsive energy: sum of 1/d^2 for all pairs, where d is angular distance
        # This encourages uniform distribution
        repulsive = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.arccos(np.clip(np.dot(pts[i], pts[j]), -1.0, 1.0))
                if d > 1e-6:
                    repulsive += 1.0 / (d * d)
        
        # Weight the repulsive term to be smaller than the covering radius term
        return log_sum + 0.01 * repulsive

    def optimize_worst_facet(pts, step_size, iterations=50):
        """
        Iteratively move the vertices of the worst facet towards the worst center.
        """
        pts = pts.copy()
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(pts)

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

            # Move vertices towards worst_center
            for idx in target_indices:
                v = pts[idx]
                dir_vec = worst_center - v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    pts[idx] = v + step_size * dir_vec
                    pts[idx] /= np.linalg.norm(pts[idx])

        return pts

    def optimize_with_repulsion(pts, step_size, iterations=50):
        """
        Optimize using repulsive forces between all points, weighted by facet importance.
        Points near worst facets get stronger repulsion from other points.
        """
        pts = pts.copy()
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(pts)
            facets = compute_facet_circumcenters(pts)
            
            # Compute facet importance (normalized radius)
            if not facets:
                break
            max_facet_r = max(r for r, _, _ in facets)
            if max_facet_r < 1e-10:
                break
            
            # For each point, compute how much it contributes to worst facets
            point_weight = np.zeros(N)
            for r, c, simplex in facets:
                if simplex is None:
                    continue
                weight = (r / max_facet_r) ** 2  # Emphasize worse facets
                for idx in simplex:
                    point_weight[idx] += weight
            
            # Normalize weights
            max_w = np.max(point_weight)
            if max_w > 0:
                point_weight /= max_w

            # Compute repulsive forces
            forces = np.zeros((N, 3))
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.arccos(np.clip(np.dot(pts[i], pts[j]), -1.0, 1.0))
                    if d < 1e-6:
                        continue
                    # Inverse square repulsion, weighted by point importance
                    force_mag = 1.0 / (d * d)
                    # Direction: from j to i for point i, from i to j for point j
                    dir_vec = pts[i] - pts[j]
                    norm_dir = np.linalg.norm(dir_vec)
                    if norm_dir > 1e-10:
                        dir_vec /= norm_dir
                        forces[i] += force_mag * point_weight[i] * dir_vec
                        forces[j] += force_mag * point_weight[j] * (-dir_vec)

            # Apply forces with step size
            for i in range(N):
                v = pts[i]
                new_v = v + step_size * forces[i]
                new_norm = np.linalg.norm(new_v)
                if new_norm > 1e-10:
                    pts[i] = new_v / new_norm

        return pts

    # Phase 1: Initial Fibonacci spiral with multiple offsets
    best = None
    best_radius = np.pi

    for offset in [0.5, 0.3, 0.7]:
        if time.time() > deadline - 10:
            break

        current = fibonacci_sphere(N, offset)

        # Phase 2: L-BFGS-B optimization on the smooth soft-max surrogate
        if time.time() < deadline - 15:
            flat_initial = current.flatten()
            try:
                result = minimize(
                    soft_max_radius_objective,
                    flat_initial,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.success:
                    current = result.x.reshape(N, 3)
            except:
                pass

        # Phase 3: Worst-facet optimization to refine
        if time.time() < deadline - 10:
            for step in [0.05, 0.02, 0.01]:
                if time.time() > deadline - 10:
                    break
                current = optimize_worst_facet(current, step, iterations=50)

        radius, _ = compute_covering_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()

    # Phase 4: Repulsive force optimization
    if time.time() < deadline - 15:
        current = best.copy()
        for step in [0.02, 0.01, 0.005]:
            if time.time() > deadline - 15:
                break
            current = optimize_with_repulsion(current, step, iterations=80)
            radius, _ = compute_covering_radius(current)
            if radius < best_radius:
                best_radius = radius
                best = current.copy()
            else:
                current = best.copy()

    # Phase 5: Random restarts with small perturbations
    if time.time() < deadline - 15:
        for restart in range(15):
            if time.time() > deadline - 15:
                break

            # Create a perturbed version of the best solution
            perturbed = best.copy()
            noise_scale = 0.04 / (restart + 1)
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])

            # Optimize the perturbed solution using soft-max then worst-facet
            if time.time() < deadline - 15:
                flat_pert = perturbed.flatten()
                try:
                    result = minimize(
                        soft_max_radius_objective,
                        flat_pert,
                        method='L-BFGS-B',
                        options={'maxiter': 150, 'ftol': 1e-9, 'gtol': 1e-7}
                    )
                    if result.success:
                        perturbed = result.x.reshape(N, 3)
                except:
                    pass

            for step in [0.02, 0.01]:
                if time.time() > deadline - 15:
                    break
                perturbed = optimize_worst_facet(perturbed, step, iterations=40)

            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 6: Final fine-tuning with very small steps
    if time.time() < deadline - 5:
        final = best.copy()
        for _ in range(80):
            if time.time() > deadline - 5:
                break
            radius, worst_center = compute_covering_radius(final)

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
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    final[idx] = v + 0.002 * dir_vec
                    final[idx] /= np.linalg.norm(final[idx])

        radius, _ = compute_covering_radius(final)
        if radius < best_radius:
            best = final

    # Final normalization and validation
    best = best / np.linalg.norm(best, axis=1, keepdims=True)

    # Ensure all points are non-zero and finite
    norms = np.linalg.norm(best, axis=1)
    valid = (norms > 1e-10) & np.all(np.isfinite(best), axis=1)
    if not np.all(valid):
        # Fallback to Fibonacci if something went wrong
        best = fibonacci_sphere(N)

    return best
# EVOLVE-BLOCK-END
