"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit

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

    def repulsion_relax(pts, n_iters=200, alpha_start=0.05):
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

    def lloyd_iteration(pts, n_iters=100, time_budget=15.0):
        """
        Weighted Lloyd iteration: move each point toward weighted average of
        circumcenters of facets it belongs to, weighted by facet radius.
        """
        pts = pts.copy()
        start_t = time.time()
        
        for it in range(n_iters):
            if time.time() - start_t > time_budget:
                break
            
            hull, facets, centers, radii, worst_idx = compute_facet_data(pts)
            if hull is None:
                break
            
            max_rad = np.max(radii)
            if max_rad < 1e-10:
                break
            
            updates = np.zeros_like(pts)
            weights_sum = np.zeros(N)
            
            for f in range(len(facets)):
                rad = radii[f]
                if rad < 1e-12:
                    continue
                # Weight: higher for larger facets
                w = (rad / max_rad) ** 2
                cc = centers[f]
                for idx in facets[f]:
                    updates[idx] += w * (cc - pts[idx])
                    weights_sum[idx] += w
            
            mask = weights_sum > 1e-10
            if np.any(mask):
                step_size = 0.03
                pts[mask] += step_size * updates[mask] / weights_sum[mask, np.newaxis]
                pts = project_to_sphere(pts)
        
        return pts

    def soft_max_radius_and_grad(points, beta=50.0):
        """
        Compute soft-max radius and its gradient w.r.t. points.
        """
        hull, facets, centers, radii, worst_idx = compute_facet_data(points)
        if hull is None or worst_idx is None:
            return np.pi, np.zeros_like(points)
        
        max_r = np.max(radii)
        # Use log-sum-exp trick for numerical stability
        shifted = beta * (radii - max_r)
        max_shifted = np.max(shifted)
        exp_terms = np.exp(shifted - max_shifted)
        log_sum = max_shifted + np.log(np.sum(exp_terms))
        soft_radius = max_r + log_sum / beta
        weights = exp_terms / np.sum(exp_terms)
        
        grads = np.zeros_like(points)
        n_facets = len(facets)
        
        for i in range(n_facets):
            w = weights[i]
            if w < 1e-15:
                continue
            r = radii[i]
            if r < 1e-12:
                continue
            
            C = centers[i]
            sin_r = np.sin(r)
            if sin_r < 1e-10:
                continue
            
            simplex = facets[i]
            p_pts = points[simplex]
            
            # Gradient of arccos(dot(C, p)) w.r.t. p is -C/sin(r)
            grad_contrib = -C / sin_r
            
            # Project to tangent space at p
            dots = np.sum(p_pts * grad_contrib, axis=1)
            radial_comps = dots[:, np.newaxis] * p_pts
            tangential_grads = grad_contrib - radial_comps
            
            for k in range(3):
                grads[simplex[k]] += w * tangential_grads[k]
        
        return soft_radius, grads

    def facet_targeted_optimization(pts, time_budget):
        """
        Structural change: Directly optimize the worst facets by moving vertices
        towards circumcenters. This avoids the smooth approximation bottleneck.
        """
        current = pts.copy()
        best = pts.copy()
        best_radius, _ = compute_covering_radius(best)
        start_t = time.time()
        
        step_size = 0.1
        min_step = 1e-5
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() - start_t < time_budget:
            hull, facets, centers, radii, worst_idx = compute_facet_data(current)
            if hull is None or worst_idx is None:
                break
                
            current_radius = radii[worst_idx]
            
            # If we are at a local minimum for this step size, reduce step
            if current_radius >= best_radius - 1e-10:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    step_size *= 0.5
                    no_improve_count = 0
                    if step_size < min_step:
                        # Try a random restart/jitter to escape local minimum
                        if time.time() - start_t > time_budget - 5:
                            break
                        current = current.copy()
                        for i in range(N):
                            jitter = np.random.randn(3) * 0.01
                            current[i] = current[i] + jitter
                            current[i] /= np.linalg.norm(current[i])
                        continue
                else:
                    # Try smaller step
                    step_size *= 0.7
                    if step_size < min_step:
                        step_size = min_step
                    continue
            
            # Accept improvement
            if current_radius < best_radius:
                best_radius = current_radius
                best = current.copy()
                no_improve_count = 0
                
            # Identify worst facet
            worst_facet = facets[worst_idx]
            worst_center = centers[worst_idx]
            
            # Generate candidate moves
            candidates = []
            
            # 1. Move all 3 vertices towards center
            c1 = current.copy()
            for idx in worst_facet:
                v = c1[idx]
                dir_vec = worst_center - v
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    c1[idx] = v + step_size * dir_vec
                    c1[idx] /= np.linalg.norm(c1[idx])
            candidates.append(c1)
            
            # 2. Move individual vertices towards center
            for k in range(3):
                idx = worst_facet[k]
                c2 = current.copy()
                v = c2[idx]
                dir_vec = worst_center - v
                radial_comp = np.dot(dir_vec, v)
                dir_vec = dir_vec - radial_comp * v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    c2[idx] = v + step_size * dir_vec
                    c2[idx] /= np.linalg.norm(c2[idx])
                candidates.append(c2)
                
            # 3. Move vertices away from center (if center is "inside" the cap, we might be over-correcting?)
            # Actually, moving towards center shrinks the facet. Moving away expands it.
            # Let's try a "push" move: move vertices slightly away from each other?
            # Or just try a random jitter on the worst facet vertices.
            c3 = current.copy()
            for idx in worst_facet:
                jitter = np.random.randn(3) * 0.005
                c3[idx] = c3[idx] + jitter
                c3[idx] /= np.linalg.norm(c3[idx])
            candidates.append(c3)
            
            # Evaluate candidates
            best_candidate = None
            best_candidate_radius = current_radius
            
            for cand in candidates:
                # Quick check: does it improve the worst facet?
                h, f, c, r, wi = compute_facet_data(cand)
                if h is None:
                    continue
                # Check if the specific worst facet is smaller
                # Note: The index of the worst facet might change, so we check the max radius
                cand_radius = np.max(r)
                if cand_radius < best_candidate_radius:
                    best_candidate_radius = cand_radius
                    best_candidate = cand
            
            if best_candidate is not None:
                current = best_candidate
            else:
                # No move improved the max radius. Reduce step size.
                step_size *= 0.8
                if step_size < min_step:
                    step_size = min_step
                    # If stuck, try a larger random kick
                    if np.random.rand() < 0.1:
                        current = current.copy()
                        for i in range(N):
                            jitter = np.random.randn(3) * 0.02
                            current[i] = current[i] + jitter
                            current[i] /= np.linalg.norm(current[i])

        return best, best_radius

    # Phase 1: Initial Fibonacci spiral
    current = fibonacci_sphere(N)
    best = current.copy()
    best_radius, _ = compute_covering_radius(best)

    # Phase 2: Repulsion relaxation to spread points
    if time.time() < deadline - 60:
        try:
            relaxed = repulsion_relax(current, n_iters=150, alpha_start=0.04)
            rel_radius, _ = compute_covering_radius(relaxed)
            if rel_radius < best_radius:
                best_radius = rel_radius
                best = relaxed.copy()
                current = relaxed.copy()
        except Exception:
            pass

    # Phase 3: Lloyd iteration for uniform spreading
    if time.time() < deadline - 50:
        try:
            lloyd_pts = lloyd_iteration(current, n_iters=80, time_budget=12.0)
            lloyd_radius, _ = compute_covering_radius(lloyd_pts)
            if lloyd_radius < best_radius:
                best_radius = lloyd_radius
                best = lloyd_pts.copy()
                current = lloyd_pts.copy()
        except Exception:
            pass

    # Phase 4: Facet-Targeted Optimization (Structural Change)
    if time.time() < deadline - 10:
        try:
            ft_pts, ft_radius = facet_targeted_optimization(current, time_budget=40.0)
            if ft_radius < best_radius:
                best_radius = ft_radius
                best = ft_pts.copy()
                current = ft_pts.copy()
        except Exception:
            pass

    # Phase 5: Second Round of Facet-Targeted Optimization with smaller initial step
    if time.time() < deadline - 5:
        try:
            # Perturb slightly to escape local minima
            perturbed = best.copy()
            for i in range(N):
                jitter = np.random.randn(3) * 0.005
                perturbed[i] = perturbed[i] + jitter
                perturbed[i] /= np.linalg.norm(perturbed[i])
            
            ft_pts2, ft_radius2 = facet_targeted_optimization(perturbed, time_budget=10.0)
            if ft_radius2 < best_radius:
                best_radius = ft_radius2
                best = ft_pts2.copy()
        except Exception:
            pass

    # Final normalization and validation
    best = project_to_sphere(best)
    norms = np.linalg.norm(best, axis=1)
    valid = (norms > 1e-10) & np.all(np.isfinite(best), axis=1)
    if not np.all(valid):
        best = fibonacci_sphere(N)

    return best
# EVOLVE-BLOCK-END
