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
        Compute all facet circumcenters and their angular radii (vectorized).
        Returns: (radii_array, centers_array, simplices_array)
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return np.array([]), np.array([]), np.array([])
        
        simplices = hull.simplices  # shape (F, 3)
        F = len(simplices)
        
        v1 = pts[simplices[:, 0]]
        v2 = pts[simplices[:, 1]]
        v3 = pts[simplices[:, 2]]
        
        edge1 = v2 - v1
        edge2 = v3 - v1
        normals = np.cross(edge1, edge2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        valid = (norms[:, 0] > 1e-12)
        if not np.any(valid):
            return np.array([]), np.array([]), np.array([])
        
        normals = normals[valid]
        norms = norms[valid]
        v1 = v1[valid]
        simplices = simplices[valid]
        
        centers = normals / norms
        dots = np.sum(centers * v1, axis=1)
        centers[dots < 0] *= -1
        
        diffs = centers - v1
        d = np.linalg.norm(diffs, axis=1)
        radii = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
        
        return radii, centers, simplices

    def compute_covering_radius(pts):
        """Compute the covering radius in radians and the worst-case circumcenter."""
        radii, centers, simplices = compute_facet_circumcenters(pts)
        if len(radii) == 0:
            return np.pi, np.array([0.0, 0.0, 1.0])
        
        max_idx = np.argmax(radii)
        return radii[max_idx], centers[max_idx]

    def spectral_energy_and_grad(pts):
        """
        Compute spectral energy (variance of pairwise distances) and its gradient.
        Minimizing this promotes uniform distribution on the sphere.
        
        The spectral energy is: E = sum_{i<j} (d_ij - d_mean)^2 / (N*(N-1))
        where d_ij is the angular distance between points i and j.
        
        For efficiency, we work with dot products: cos(d_ij) = p_i . p_j
        """
        n = len(pts)
        # Compute all pairwise dot products
        dots = np.dot(pts, pts.T)  # (n, n)
        
        # Mean dot product (excluding diagonal)
        mask = ~np.eye(n, dtype=bool)
        mean_dot = np.sum(dots[mask]) / (n * (n - 1))
        
        # Energy: sum of squared deviations from mean
        # E = sum_{i<j} (dot_ij - mean_dot)^2
        # For gradient, we need dE/dp_k
        # d(dot_ij)/dp_k = delta_ik * p_j + delta_jk * p_i for i != j
        
        # Compute gradient
        grad = np.zeros_like(pts)
        
        # For each point k, the contribution to the gradient is:
        # sum_{j != k} 2 * (dot_kj - mean_dot) * (p_j - (dot_kj - mean_dot) * p_k)
        # Wait, let me think more carefully.
        
        # E = sum_{i<j} (dot_ij - mean)^2
        # dE/dp_k = 2 * sum_{j != k, j > k} (dot_kj - mean) * (p_j - (dot_kj) * p_k)
        #          + 2 * sum_{i != k, i < k} (dot_ik - mean) * (p_i - (dot_ik) * p_k)
        #          - 2 * (dot_kj - mean) * p_k * (1/(n-1)) for the mean term
        # This is getting complex. Let me use a simpler formulation.
        
        # Alternative: E = sum_{i,j} (dot_ij - mean)^2 / 2 (with i != j)
        # dE/dp_k = sum_{j != k} (dot_kj - mean) * d(dot_kj)/dp_k
        #         = sum_{j != k} (dot_kj - mean) * (p_j - (dot_kj) * p_k)
        #         - sum_{j != k} (dot_kj - mean) * p_k * (1/(n-1)) * (n-1)  [from mean derivative]
        # Actually, the mean depends on all pairs, so:
        # d(mean)/dp_k = (1/(n*(n-1))) * sum_{j != k} (p_j + p_k * 0)  [since dot_kj = p_k.p_j]
        #              = (1/(n*(n-1))) * sum_{j != k} p_j
        #              = (1/(n*(n-1))) * (sum_all_p - p_k)
        
        sum_all_p = np.sum(pts, axis=0)
        dmean_dpk = (sum_all_p - pts[k]) / (n * (n - 1)) if False else None  # per-point
        
        # Let me just compute numerically for clarity, then vectorize
        # Actually, let's use a cleaner derivation:
        
        # E = (1/(n*(n-1))) * sum_{i<j} (dot_ij - mean_dot)^2
        # Let S = sum_{i<j} (dot_ij - mean_dot)^2
        # dS/dp_k = 2 * sum_{j != k} (dot_kj - mean_dot) * (p_j - (dot_kj) * p_k)
        #          - 2 * (sum_{j != k} (dot_kj - mean_dot)) * (sum_all_p - p_k) / (n*(n-1))
        #          - 2 * sum_{j != k} (dot_kj - mean_dot) * p_k * (1/(n-1))  [from mean in each term]
        
        # This is getting messy. Let me use a different approach:
        # Minimize the variance of cosines, which is equivalent to maximizing the trace of G^2
        # where G is the Gram matrix.
        
        # Simpler: minimize sum_{i,j} (dot_ij)^2 - (sum_{i,j} dot_ij)^2 / (n^2)
        # This is the variance of all pairwise dot products (including diagonal)
        
        # Even simpler and more standard: minimize the "energy" E = sum_{i<j} (1 - dot_ij^2)
        # which is related to the sum of squared sines of angles.
        
        # Let me use the standard Riesz s-energy with s=2:
        # E = sum_{i<j} (1 - dot_ij^2) = sum_{i<j} sin^2(d_ij)
        # This is smooth and promotes uniform distribution.
        
        # Gradient of E w.r.t. p_k:
        # dE/dp_k = sum_{j != k} 2 * (1 - dot_kj^2) * (-dot_kj) * (p_j - dot_kj * p_k) / 2
        #         = sum_{j != k} (dot_kj^2 - 1) * dot_kj * (p_j - dot_kj * p_k)
        #         = sum_{j != k} dot_kj * (dot_kj^2 - 1) * p_j - sum_{j != k} dot_kj^2 * (dot_kj^2 - 1) * p_k
        
        # Let me just compute it directly
        dots_k = dots[k, :]  # (n,)
        dots_k[k] = 0  # exclude self
        
        # E = sum_{j != k} (1 - dots_kj^2) for each k, but we count each pair once
        # Total E = 0.5 * sum_k sum_{j != k} (1 - dots_kj^2)
        
        # dE/dp_k = sum_{j != k} (dots_kj^2 - 1) * dots_kj * (p_j - dots_kj * p_k)
        
        terms = dots_k * (dots_k**2 - 1)  # (n,)
        grad_k = np.sum(pts * terms[:, np.newaxis], axis=0) - np.sum(terms * dots_k) * pts[k]
        
        # Project to tangent space
        grad_k = grad_k - np.dot(grad_k, pts[k]) * pts[k]
        grad[k] = grad_k
        
        # Compute total energy
        E = 0.5 * np.sum((1 - dots[mask]**2))
        
        return E, grad

    def soft_max_radius_and_grad(pts, beta=50.0):
        """
        Compute soft-max radius and its gradient w.r.t. points using analytical gradients.
        Returns: (soft_radius, gradient_array)
        """
        radii, centers, simplices = compute_facet_circumcenters(pts)
        if len(radii) == 0:
            return np.pi, np.zeros_like(pts)
        
        max_r = np.max(radii)
        shifted = beta * (radii - max_r)
        max_shifted = np.max(shifted)
        exp_terms = np.exp(shifted - max_shifted)
        log_sum = max_shifted + np.log(np.sum(exp_terms))
        soft_radius = max_r + log_sum / beta
        weights = exp_terms / np.sum(exp_terms)
        
        grads = np.zeros_like(pts)
        F = len(radii)
        
        for i in range(F):
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
            
            simplex = simplices[i]
            p_pts = pts[simplex]
            
            grad_contrib = -C / sin_r
            
            dots = np.sum(p_pts * grad_contrib, axis=1)
            radial_comps = dots[:, np.newaxis] * p_pts
            tangential_grads = grad_contrib - radial_comps
            
            for k in range(3):
                grads[simplex[k]] += w * tangential_grads[k]
        
        return soft_radius, grads

    def project_to_sphere(pts):
        """Project points onto the unit sphere."""
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return pts / norms

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

    # Phase 1: Generate multiple diverse initial configurations
    initial_configs = []
    
    # Fibonacci spiral
    initial_configs.append(fibonacci_sphere(N))
    
    # Fibonacci with different offset
    initial_configs.append(fibonacci_sphere(N, offset=0.3))
    
    # Random configuration (seeded for reproducibility)
    np.random.seed(123)
    rand_pts = np.random.randn(N, 3)
    rand_pts /= np.linalg.norm(rand_pts, axis=1, keepdims=True)
    initial_configs.append(rand_pts)
    
    # Another random with different seed
    np.random.seed(456)
    rand_pts2 = np.random.randn(N, 3)
    rand_pts2 /= np.linalg.norm(rand_pts2, axis=1, keepdims=True)
    initial_configs.append(rand_pts2)
    
    # Evaluate all initial configs
    best = None
    best_radius = np.pi
    for config in initial_configs:
        radius, _ = compute_covering_radius(config)
        if radius < best_radius:
            best_radius = radius
            best = config.copy()
    
    current = best.copy()

    # Phase 2: Spectral energy minimization (smooth surrogate)
    if time.time() < deadline - 50:
        try:
            x0 = current.flatten()
            
            def spectral_obj_and_grad(x):
                if time.time() > deadline - 40:
                    return 1e10, np.zeros_like(x)
                pts = x.reshape(N, 3)
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms = np.where(norms < 1e-10, 1.0, norms)
                pts = pts / norms
                energy, grad = spectral_energy_and_grad(pts)
                return energy, grad.reshape(-1)
            
            elapsed = time.time() - start_time
            remaining = deadline - 50 - elapsed
            if remaining > 10:
                max_iter = int(min(1000, max(200, 300 * (remaining / 30.0))))
                
                result = minimize(
                    spectral_obj_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'maxfun': max_iter*3, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 200}
                )
                
                if result.success or result.fun < spectral_obj_and_grad(x0)[0]:
                    spectral_pts = project_to_sphere(result.x.reshape(N, 3))
                    spectral_radius, _ = compute_covering_radius(spectral_pts)
                    if spectral_radius < best_radius:
                        best_radius = spectral_radius
                        best = spectral_pts.copy()
                        current = spectral_pts.copy()
        except Exception:
            pass

    # Phase 3: L-BFGS-B with soft-max covering radius (progressive beta)
    if time.time() < deadline - 30:
        try:
            x0 = current.flatten()
            beta_schedule = [30.0, 80.0, 150.0, 300.0, 600.0]
            for beta in beta_schedule:
                if time.time() > deadline - 30:
                    break
                
                def objective_and_grad(x, b=beta):
                    if time.time() > deadline - 20:
                        return 1e10, np.zeros_like(x)
                    pts = x.reshape(N, 3)
                    norms = np.linalg.norm(pts, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-10, 1.0, norms)
                    pts = pts / norms
                    soft_r, grads = soft_max_radius_and_grad(pts, beta=b)
                    return soft_r, grads.reshape(-1)
                
                elapsed = time.time() - start_time
                remaining = deadline - 30 - elapsed
                if remaining < 5:
                    break
                max_iter = int(min(500, max(100, 150 * (remaining / 30.0))))
                
                result = minimize(
                    objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'maxfun': max_iter*3, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 200}
                )
                
                if result.success or result.fun < objective_and_grad(x0)[0]:
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
        if time.time() > deadline - 20:
            break
        current = optimize_worst_facet(current, step, iterations=100)
        radius, _ = compute_covering_radius(current)
        if radius < best_radius:
            best_radius = radius
            best = current.copy()
        else:
            current = best.copy()

    # Phase 5: Random restarts with spectral energy refinement
    if time.time() < deadline - 30:
        np.random.seed(42)
        for restart in range(60):
            if time.time() > deadline - 30:
                break
            noise_scale = 0.05 * (0.7 ** (restart // 10))
            perturbed = best.copy()
            for i in range(N):
                rand_vec = np.random.randn(3)
                rand_vec /= np.linalg.norm(rand_vec)
                perturbed[i] = perturbed[i] + noise_scale * rand_vec
                perturbed[i] /= np.linalg.norm(perturbed[i])
            
            # Quick spectral energy refinement
            if time.time() < deadline - 30:
                try:
                    x0 = perturbed.flatten()
                    
                    def quick_spectral_obj_and_grad(x):
                        if time.time() > deadline - 30:
                            return 1e10, np.zeros_like(x)
                        pts = x.reshape(N, 3)
                        norms = np.linalg.norm(pts, axis=1, keepdims=True)
                        norms = np.where(norms < 1e-10, 1.0, norms)
                        pts = pts / norms
                        energy, grad = spectral_energy_and_grad(pts)
                        return energy, grad.reshape(-1)
                    
                    result = minimize(
                        quick_spectral_obj_and_grad,
                        x0,
                        method='L-BFGS-B',
                        jac=True,
                        options={'maxiter': 100, 'maxfun': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 100}
                    )
                    
                    if result.success:
                        refined_pts = project_to_sphere(result.x.reshape(N, 3))
                        radius, _ = compute_covering_radius(refined_pts)
                        if radius < best_radius:
                            best_radius = radius
                            best = refined_pts.copy()
                            current = refined_pts.copy()
                except Exception:
                    pass
            
            # Also try worst-facet optimization
            for step in [0.02, 0.01, 0.005, 0.002, 0.001]:
                if time.time() > deadline - 30:
                    break
                perturbed = optimize_worst_facet(perturbed, step, iterations=80)
            radius, _ = compute_covering_radius(perturbed)
            if radius < best_radius:
                best_radius = radius
                best = perturbed.copy()

    # Phase 6: Final high-precision polish
    if time.time() < deadline - 5:
        final = best.copy()
        beta_final = 800.0
        
        def final_objective_and_grad(x, b=beta_final):
            if time.time() > deadline - 3:
                return 1e10, np.zeros_like(x)
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            pts = pts / norms
            soft_r, grads = soft_max_radius_and_grad(pts, beta=b)
            return soft_r, grads.reshape(-1)
        
        try:
            x0 = final.flatten()
            remaining = deadline - 5 - (time.time() - start_time)
            if remaining > 5:
                max_iter = int(min(800, max(200, 250 * (remaining / 10.0))))
                result = minimize(
                    final_objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'maxfun': max_iter*3, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 300}
                )
                if result.success:
                    polish_pts = project_to_sphere(result.x.reshape(N, 3))
                    polish_radius, _ = compute_covering_radius(polish_pts)
                    if polish_radius < best_radius:
                        best_radius = polish_radius
                        best = polish_pts.copy()
        except Exception:
            pass

        # Additional manual polish with tiny steps
        if time.time() < deadline - 2:
            final = best.copy()
            step = 0.001
            prev_radius = compute_covering_radius(final)[0]
            stall_count = 0
            for it in range(5000):
                if time.time() > deadline - 2:
                    break
                radius, worst_center = compute_covering_radius(final)
                if radius >= prev_radius - 1e-16:
                    step *= 0.7
                    stall_count += 1
                    if step < 1e-16 or stall_count > 150:
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
