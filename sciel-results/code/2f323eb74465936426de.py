"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def compute_facet_circumcenters(pts):
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

    def epigraph_objective(flat_params, beta=80.0):
        """
        Epigraph formulation: minimize t subject to all facet radii <= t.
        Uses soft constraints via log-sum-exp for smoothness.
        flat_params = [point_1_x, point_1_y, point_1_z, ..., point_50_x, point_50_y, point_50_z, t]
        """
        pts = flat_params[:N*3].reshape(N, 3)
        t = flat_params[N*3]
        
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms

        facets = compute_facet_circumcenters(pts)
        if not facets or len(facets) < 2:
            return t + 10.0

        radii = np.array([r for r, _, _ in facets])
        
        # Soft constraint: sum of exp(beta * (radius - t)) should be small
        # This is smooth and differentiable
        diff = radii - t
        max_diff = np.max(diff)
        if max_diff > 0:
            log_sum = max_diff + np.log(np.sum(np.exp((diff - max_diff) * beta)))
            penalty = log_sum / beta
        else:
            penalty = 0.0
        
        # Objective: t + penalty (penalty pushes t up if any radius > t)
        return t + penalty

    def optimize_worst_facet(pts, step_size, iterations=50):
        pts = pts.copy()
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(pts)

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

            for idx in target_indices:
                v = pts[idx]
                dir_vec = worst_center - v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    pts[idx] = v + step_size * dir_vec
                    pts[idx] /= np.linalg.norm(pts[idx])

        return pts

    def lloyd_iteration(pts, iterations=30):
        """
        Lloyd iteration: for each facet, move its vertices toward the facet's
        circumcenter to reduce the covering radius.
        """
        pts = pts.copy()
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break

            facets = compute_facet_circumcenters(pts)
            if not facets:
                break

            # Compute target displacement for each point
            displacement = np.zeros((N, 3))
            count = np.zeros(N)
            for radius, center, simplex in facets:
                if simplex is None:
                    continue
                # Weight by radius to prioritize larger facets
                weight = radius / max(f[0] for f in facets)
                for idx in simplex:
                    displacement[idx] += weight * (center - pts[idx])
                    count[idx] += 1

            # Apply averaged displacement
            for i in range(N):
                if count[i] > 0:
                    disp = displacement[i] / count[i]
                    pts[i] = pts[i] + 0.1 * disp
                    pts[i] /= np.linalg.norm(pts[i])

        return pts

    def repulsion_relax(pts, iterations=50, strength=0.01):
        """
        Relax points using pairwise repulsion to spread them more uniformly.
        """
        pts = pts.copy()
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break

            forces = np.zeros((N, 3))
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.arccos(np.clip(np.dot(pts[i], pts[j]), -1.0, 1.0))
                    if d < 1e-6:
                        continue
                    force_mag = strength / (d * d)
                    dir_vec = pts[i] - pts[j]
                    norm_dir = np.linalg.norm(dir_vec)
                    if norm_dir > 1e-10:
                        dir_vec /= norm_dir
                        forces[i] += force_mag * dir_vec
                        forces[j] -= force_mag * dir_vec

            for i in range(N):
                new_v = pts[i] + 0.05 * forces[i]
                new_norm = np.linalg.norm(new_v)
                if new_norm > 1e-10:
                    pts[i] = new_v / new_norm

        return pts

    # Phase 1: Initial Fibonacci spiral
    best = fibonacci_sphere(N, 0.5)
    best_radius = compute_covering_radius(best)[0]

    # Phase 2: Pre-processing with Lloyd and repulsion
    current = best.copy()
    if time.time() < deadline - 20:
        current = lloyd_iteration(current, iterations=30)
        radius, _ = compute_covering_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()

    if time.time() < deadline - 20:
        current = repulsion_relax(best.copy(), iterations=50)
        radius, _ = compute_covering_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()

    # Phase 3: L-BFGS-B optimization using epigraph formulation
    if time.time() < deadline - 25:
        # Initialize with current best points and t = current covering radius + small margin
        initial_radius = compute_covering_radius(best)[0]
        t_init = initial_radius + 0.01
        flat_initial = np.concatenate([best.flatten(), [t_init]])
        
        try:
            result = minimize(
                epigraph_objective,
                flat_initial,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
            )
            if result.success and result.x is not None:
                current = result.x[:N*3].reshape(N, 3)
                # Normalize to ensure valid output
                current = current / np.linalg.norm(current, axis=1, keepdims=True)
                radius, _ = compute_covering_radius(current)
                if radius < best_radius:
                    best_radius = radius
                    best = current.copy()
        except:
            pass

    # Phase 4: Worst-facet optimization
    if time.time() < deadline - 15:
        for step in [0.05, 0.02, 0.01]:
            if time.time() > deadline - 15:
                break
            current = optimize_worst_facet(best.copy(), step, iterations=50)
            radius, _ = compute_covering_radius(current)
            if radius < best_radius:
                best_radius = radius
                best = current.copy()

    # Phase 5: Random restarts with perturbations
    if time.time() < deadline - 20:
        for restart in range(12):
            if time.time() > deadline - 20:
                break

            perturbed = best.copy()
            noise_scale = 0.04 / (restart + 1)
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])

            # Apply Lloyd then L-BFGS
            if time.time() < deadline - 20:
                perturbed = lloyd_iteration(perturbed, iterations=20)

            if time.time() < deadline - 20:
                # Use epigraph formulation for restarts too
                pert_radius = compute_covering_radius(perturbed)[0]
                t_pert = pert_radius + 0.01
                flat_pert = np.concatenate([perturbed.flatten(), [t_pert]])
                try:
                    result = minimize(
                        epigraph_objective,
                        flat_pert,
                        method='L-BFGS-B',
                        options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 15}
                    )
                    if result.success and result.x is not None:
                        perturbed = result.x[:N*3].reshape(N, 3)
                        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
                except:
                    pass

            if time.time() < deadline - 20:
                perturbed = optimize_worst_facet(perturbed, 0.01, iterations=30)

            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 6: Final fine-tuning with high-precision epigraph polish
    if time.time() < deadline - 5:
        # First do a few steps of worst-facet local search
        final = best.copy()
        for _ in range(30):
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

        # Then do a high-precision epigraph polish
        if time.time() < deadline - 3:
            final_radius = compute_covering_radius(final)[0]
            t_final = final_radius + 0.005
            flat_final = np.concatenate([final.flatten(), [t_final]])
            try:
                result = minimize(
                    epigraph_objective,
                    flat_final,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
                )
                if result.success and result.x is not None:
                    final = result.x[:N*3].reshape(N, 3)
                    final = final / np.linalg.norm(final, axis=1, keepdims=True)
            except:
                pass

        radius, _ = compute_covering_radius(final)
        if radius < best_radius:
            best = final

    # Final normalization and validation
    best = best / np.linalg.norm(best, axis=1, keepdims=True)

    norms = np.linalg.norm(best, axis=1)
    valid = (norms > 1e-10) & np.all(np.isfinite(best), axis=1)
    if not np.all(valid):
        best = fibonacci_sphere(N)

    return best
# EVOLVE-BLOCK-END
