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

    def optimize_lbfgs_schedule(pts, time_budget):
        """
        Run L-BFGS-B with a schedule of increasing beta values.
        """
        current = pts.copy()
        best = pts.copy()
        best_radius, _ = compute_covering_radius(best)
        start_t = time.time()
        
        # Schedule of beta values: start smooth, end sharp
        beta_schedule = [10.0, 50.0, 200.0, 800.0]
        
        for beta in beta_schedule:
            if time.time() - start_t > time_budget:
                break
                
            def obj_grad(x, b=beta):
                if time.time() - start_t > time_budget:
                    raise Exception("timeout")
                p = project_to_sphere(x.reshape(N, 3))
                soft_r, grads = soft_max_radius_and_grad(p, beta=b)
                return soft_r, grads.reshape(-1)

            try:
                x0 = current.flatten()
                remaining = time_budget - (time.time() - start_t)
                max_iter = int(min(150, max(40, 8 * remaining)))
                
                result = minimize(
                    obj_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'maxfun': max_iter*2, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if np.isfinite(result.fun):
                    new_pts = project_to_sphere(result.x.reshape(N, 3))
                    new_radius, _ = compute_covering_radius(new_pts)
                    
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best = new_pts.copy()
                        current = new_pts.copy()
                        
            except Exception:
                pass
                
        return best, best_radius

    def polish_gradient_descent(pts, time_budget, beta=500.0):
        """
        Final local polish using projected gradient descent on tangent space.
        """
        current = pts.copy()
        best = pts.copy()
        best_radius, _ = compute_covering_radius(best)
        start_t = time.time()
        
        step_size = 0.008
        min_step = 1e-13
        decay = 0.75
        
        while time.time() - start_t < time_budget:
            soft_radius, grads = soft_max_radius_and_grad(current, beta=beta)
            
            grad_norm = np.linalg.norm(grads)
            if grad_norm < 1e-12:
                break
                
            new_pts = current - step_size * grads
            new_pts = project_to_sphere(new_pts)
            
            new_radius, _ = compute_covering_radius(new_pts)
            
            if new_radius < best_radius:
                best_radius = new_radius
                best = new_pts.copy()
                current = new_pts.copy()
            else:
                step_size *= decay
                if step_size < min_step:
                    break
                    
        return best, best_radius

    def optimize_worst_facet(pts, step_size, iterations=30):
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
                if step_size < 1e-9:
                    break
            else:
                prev_radius = radius

            try:
                hull, facets, centers, radii, _ = compute_facet_data(pts)
            except:
                break
            
            if hull is None:
                break

            target_indices_set = set()
            for i in range(len(facets)):
                c = centers[i]
                if np.dot(c, worst_center) > 0.999:
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

    # Phase 4: L-BFGS with Beta Schedule (Main Optimization)
    if time.time() < deadline - 35:
        try:
            lbfgs_pts, lbfgs_radius = optimize_lbfgs_schedule(current, time_budget=25.0)
            if lbfgs_radius < best_radius:
                best_radius = lbfgs_radius
                best = lbfgs_pts.copy()
                current = lbfgs_pts.copy()
        except Exception:
            pass

    # Phase 5: Projected Gradient Polish
    if time.time() < deadline - 20:
        try:
            polish_pts, polish_radius = polish_gradient_descent(current, time_budget=12.0, beta=500.0)
            if polish_radius < best_radius:
                best_radius = polish_radius
                best = polish_pts.copy()
                current = polish_pts.copy()
        except Exception:
            pass

    # Phase 6: Heuristic Worst-Facet Optimization
    step_sizes = [0.04, 0.02, 0.01, 0.005, 0.002, 0.001]
    for step in step_sizes:
        if time.time() > deadline - 12:
            break
        current = optimize_worst_facet(current, step, iterations=25)
        radius, _ = compute_covering_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()
        else:
            current = best.copy()

    # Phase 7: Random Restarts with Full Pipeline
    if time.time() < deadline - 15:
        np.random.seed(42)
        restarts = 15
        for restart in range(restarts):
            if time.time() > deadline - 10:
                break
            
            perturbed = best.copy()
            noise_scale = 0.008 / (restart + 1)
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])
            
            # Quick L-BFGS refinement
            if time.time() < deadline - 12:
                try:
                    lbfgs_pts, lbfgs_radius = optimize_lbfgs_schedule(perturbed, time_budget=2.0)
                    if lbfgs_radius < compute_covering_radius(perturbed)[0]:
                        perturbed = lbfgs_pts
                except Exception:
                    pass
            
            # Quick Gradient Polish
            if time.time() < deadline - 11:
                try:
                    polish_pts, polish_radius = polish_gradient_descent(perturbed, time_budget=1.0, beta=200.0)
                    if polish_radius < compute_covering_radius(perturbed)[0]:
                        perturbed = polish_pts
                except Exception:
                    pass
            
            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 8: Final High-Precision Polish
    if time.time() < deadline - 3:
        try:
            final_pts, final_radius = polish_gradient_descent(best, time_budget=4.0, beta=1500.0)
            if final_radius < best_radius:
                best = final_pts
                best_radius = final_radius
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
