"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

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

    def project_to_sphere(pts):
        """Project points onto the unit sphere."""
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return pts / norms

    def compute_facet_data(points):
        """
        Compute convex hull and facet circumcenters/radii (vectorized).
        Returns: hull, facets, centers (array), radii (array), worst_idx
        """
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        if np.any(norms < 1e-15):
            return None, None, None, None, None
        points = points / norms
        
        try:
            hull = ConvexHull(points)
        except Exception:
            return None, None, None, None, None
        
        facets = hull.simplices
        n_facets = len(facets)
        if n_facets == 0:
            return hull, facets, np.zeros((0,3)), np.zeros(0), None
            
        a = points[facets[:, 0]]  # (F, 3)
        b = points[facets[:, 1]]
        c = points[facets[:, 2]]
        
        # Vectorized circumcenter computation
        v1 = a - b
        v2 = b - c
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1)
        
        # Handle degenerate facets (collinear points)
        valid = norm_n > 1e-12
        
        centers = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)
        
        if np.any(valid):
            cc = n[valid] / norm_n[valid, np.newaxis]
            
            # Ensure circumcenter faces outward (toward 'a')
            dots = np.sum(cc * a[valid], axis=1)
            cc[dots < 0] *= -1
            
            centers[valid] = cc
            
            # Calculate radii (angular distance from center to vertex a)
            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)
        
        if n_facets > 0 and np.any(radii > 0):
            worst_idx = np.argmax(radii)
        else:
            worst_idx = None
            
        return hull, facets, centers, radii, worst_idx

    def compute_covering_radius(points):
        """
        Compute the actual covering radius (max facet circumradius).
        Returns radius and worst center.
        """
        hull, facets, centers, radii, worst_idx = compute_facet_data(points)
        if hull is None or worst_idx is None:
            return np.pi, np.array([0.0, 0.0, 1.0])
        return radii[worst_idx], centers[worst_idx]

    def repulsion_relax(pts, n_iters=100, alpha_start=0.05):
        """
        Repulsion relaxation: minimize sum 1/||p_i - p_j||^2 via gradient descent.
        Helps spread points more uniformly before covering optimization.
        """
        pts = pts.copy()
        alpha = alpha_start
        start_t = time.time()
        
        def repulsion_energy(p):
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            mask = ~np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
            return 0.5 * np.sum(inv_dist_sq)
        
        current_energy = repulsion_energy(pts)
        
        for it in range(n_iters):
            if time.time() > deadline - 30:
                break
            
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            
            mask = np.eye(N, dtype=bool)
            inv_dist_sq = np.zeros_like(dist_sq)
            np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
            
            forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
            
            # Project onto tangent plane
            dot_fp = np.sum(forces * pts, axis=1)
            forces_tan = forces - dot_fp[:, np.newaxis] * pts
            
            # Adaptive step with backtracking
            step = alpha
            improved = False
            for _ in range(8):
                new_pts = pts + step * forces_tan
                new_pts = project_to_sphere(new_pts)
                new_energy = repulsion_energy(new_pts)
                if new_energy < current_energy:
                    pts = new_pts
                    current_energy = new_energy
                    improved = True
                    break
                step *= 0.5
            
            if not improved:
                alpha *= 0.85
            else:
                alpha = min(alpha * 1.02, 0.08)
            
            alpha = max(alpha, 1e-6)
        
        return pts

    def lloyd_iteration(pts, n_iters=200, alpha_start=0.05, time_budget=10.0):
        """
        Weighted Lloyd iteration: move each point toward weighted average of
        circumcenters of facets it belongs to, weighted by facet radius squared.
        """
        pts = pts.copy()
        start_t = time.time()
        alpha = alpha_start
        
        for it in range(n_iters):
            if time.time() - start_t > time_budget or time.time() > deadline - 10:
                break
            
            hull, facets, centers, radii, worst_idx = compute_facet_data(pts)
            if hull is None:
                break
            
            max_rad = np.max(radii)
            if max_rad < 1e-10:
                break
            
            updates = np.zeros_like(pts)
            weights_sum = np.zeros(N)
            
            # Vectorized accumulation is hard due to variable facets per vertex,
            # so we loop over facets but use numpy ops inside
            for f in range(len(facets)):
                rad = radii[f]
                if rad < 1e-12:
                    continue
                # Weight: higher for larger facets
                w = (rad / max_rad) ** 2
                cc = centers[f]
                # Update the 3 vertices of this facet
                idxs = facets[f]
                for idx in idxs:
                    updates[idx] += w * (cc - pts[idx])
                    weights_sum[idx] += w
            
            mask = weights_sum > 1e-10
            if np.any(mask):
                # Adaptive step size
                step = alpha
                improved = False
                # Simple backtracking to ensure radius doesn't increase significantly
                current_radius, _ = compute_covering_radius(pts)
                
                for _ in range(5):
                    new_pts = pts.copy()
                    new_pts[mask] += step * updates[mask] / weights_sum[mask, np.newaxis]
                    new_pts = project_to_sphere(new_pts)
                    
                    new_radius, _ = compute_covering_radius(new_pts)
                    if new_radius < current_radius:
                        pts = new_pts
                        current_radius = new_radius
                        improved = True
                        break
                    step *= 0.5
                
                if not improved:
                    alpha *= 0.9
                else:
                    alpha = min(alpha * 1.05, 0.1)
                alpha = max(alpha, 1e-5)
        
        return pts

    def optimize_worst_facet(pts, step_size, iterations=50):
        """
        Iteratively move the vertices of the worst facet towards the worst center.
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

    # Phase 1: Initial Fibonacci spiral
    current = fibonacci_sphere(N)
    best = current.copy()
    best_radius, _ = compute_covering_radius(best)

    # Phase 2: Repulsion Relaxation to ensure good spread
    if time.time() < deadline - 100:
        try:
            current = repulsion_relax(current, n_iters=100)
            radius, _ = compute_covering_radius(current)
            if radius < best_radius:
                best_radius = radius
                best = current.copy()
        except Exception:
            pass

    # Phase 3: Lloyd's Algorithm (Main Optimization)
    if time.time() < deadline - 40:
        try:
            # Run Lloyd's with a generous time budget
            budget = min(30.0, deadline - time.time() - 20)
            current = lloyd_iteration(current, n_iters=500, alpha_start=0.05, time_budget=budget)
            radius, _ = compute_covering_radius(current)
            if radius < best_radius:
                best_radius = radius
                best = current.copy()
        except Exception:
            pass

    # Phase 4: Worst-facet optimization with adaptive step size
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    for step in step_sizes:
        if time.time() > deadline - 20:
            break
        current = optimize_worst_facet(current, step, iterations=50)
        radius, _ = compute_covering_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()
        else:
            current = best.copy()

    # Phase 5: Random restarts with Lloyd's refinement
    if time.time() < deadline - 25:
        np.random.seed(42)
        for restart in range(50):
            if time.time() > deadline - 25:
                break
            perturbed = best.copy()
            noise_scale = 0.02 / (restart + 1)
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])
            
            # Apply Lloyd's to the perturbed points
            budget = min(5.0, deadline - time.time() - 15)
            if budget > 1.0:
                try:
                    perturbed = lloyd_iteration(perturbed, n_iters=100, alpha_start=0.02, time_budget=budget)
                except Exception:
                    pass
            
            # Apply worst-facet polish
            perturbed = optimize_worst_facet(perturbed, 0.005, iterations=20)
            
            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 6: Final high-precision polish
    if time.time() < deadline - 5:
        final = best.copy()
        # Short Lloyd's pass
        budget = min(5.0, deadline - time.time() - 2)
        if budget > 0.5:
            try:
                final = lloyd_iteration(final, n_iters=100, alpha_start=0.01, time_budget=budget)
            except Exception:
                pass
        
        # Final worst-facet polish
        final = optimize_worst_facet(final, 0.0005, iterations=100)
        
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
