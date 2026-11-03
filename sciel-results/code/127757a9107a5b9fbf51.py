import numpy as np
import time
from scipy.optimize import least_squares

N = 300

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 150s kill
    
    # Precompute pair indices
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    def compute_score(pts):
        """Compute the packing radius score."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        return np.min(all_vals)
    
    def soft_min_weights(vals, k, current_min):
        """Compute soft-min weights using log-sum-exp for stability."""
        shifted = vals - current_min
        mask_close = shifted < 50.0
        if not np.any(mask_close):
            return np.ones_like(vals) / len(vals)
        close_shifted = shifted[mask_close]
        max_c = np.max(close_shifted)
        if max_c > 0:
            close_shifted = close_shifted - max_c
        exp_vals = np.exp(-k * close_shifted)
        weights = np.zeros_like(vals)
        weights[mask_close] = exp_vals
        sum_w = np.sum(weights)
        if sum_w > 1e-15:
            weights /= sum_w
        else:
            weights = np.ones_like(vals) / len(vals)
        return weights
    
    def compute_grads(pts, weights, dists, rad, bound_dists):
        """Compute gradient of soft-min objective."""
        grads = np.zeros_like(pts)
        
        # Pair contributions
        pair_weights = weights[:num_pairs]
        d_pair = dists[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-10
        if np.any(mask_nonzero):
            vecs = np.zeros((num_pairs, 2))
            idx_m = mask_nonzero
            diff_m = pts[i_idx[idx_m]] - pts[j_idx[idx_m]]
            vecs[idx_m] = (diff_m / d_pair[idx_m, np.newaxis]) * pair_weights[idx_m, np.newaxis]
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        # Boundary contributions
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-10, rad, 1.0)
        bound_grads = -(pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        return grads
    
    def optimize(pts, alpha_init, k_init, k_end, cooling, time_budget,
                 stall_threshold=150, noise_scale_base=0.01):
        """Main optimization loop using soft-min gradient ascent."""
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        alpha = alpha_init
        k = k_init
        stall_count = 0
        end_time = start_time + time_budget
        
        while time.time() < end_time:
            # Compute distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            rad = np.linalg.norm(pts, axis=1)
            bound_dists = 1.0 - rad
            
            pair_vals = dists[i_idx, j_idx] / 2.0
            all_vals = np.concatenate([pair_vals, bound_dists])
            current_min = np.min(all_vals)
            
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            # Compute weights
            weights = soft_min_weights(all_vals, k, current_min)
            
            # Compute gradients
            grads = compute_grads(pts, weights, dists, rad, bound_dists)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1.0:
                grads /= grad_norm
            
            # Update
            pts += alpha * grads
            
            # Project back inside circle
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            # Cool down
            alpha *= cooling
            
            # Increase k
            progress = min(1.0, (time.time() - start_time) / time_budget)
            k = k_init + (k_end - k_init) * (progress ** 1.5)
            
            # Random restart on stall
            if stall_count > stall_threshold:
                noise_scale = noise_scale_base * (1.0 + alpha * 10)
                pts += np.random.normal(0, noise_scale, size=pts.shape)
                rad_new = np.linalg.norm(pts, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    pts[mask_out] /= rad_new[mask_out, np.newaxis]
                stall_count = 0
                alpha = max(alpha, 0.005)
        
        return best_pts, best_score
    
    def epigraph_polish(pts, time_budget):
        """
        Polish using epigraph formulation: maximize t subject to all constraints >= t.
        Use smooth penalty: minimize sum of max(0, t - val_i)^2
        We optimize over (pts, t) to maximize t while keeping all constraints >= t.
        """
        if time.time() >= start_time + time_budget:
            return pts, compute_score(pts)
        
        best_pts = pts.copy()
        best_score = compute_score(pts)
        t = best_score * 0.9995  # Start slightly below current min
        
        def residuals(x):
            p = x[:2*N].reshape(N, 2)
            t_val = x[2*N]
            
            # Pairwise distances
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Boundary distances
            rad = np.linalg.norm(p, axis=1)
            bound_dists = 1.0 - rad
            
            pair_vals = dists[i_idx, j_idx] / 2.0
            all_vals = np.concatenate([pair_vals, bound_dists])
            
            # Residuals: max(0, t_val - val_i) for each constraint
            # Use smooth approximation: (t_val - val_i + sqrt((t_val - val_i)^2 + eps)) / 2
            eps = 1e-8
            raw = t_val - all_vals
            smooth_res = (raw + np.sqrt(raw**2 + eps)) / 2.0
            
            return smooth_res
        
        def jacobian(x):
            p = x[:2*N].reshape(N, 2)
            t_val = x[2*N]
            
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            rad = np.linalg.norm(p, axis=1)
            bound_dists = 1.0 - rad
            
            pair_vals = dists[i_idx, j_idx] / 2.0
            all_vals = np.concatenate([pair_vals, bound_dists])
            
            eps = 1e-8
            raw = t_val - all_vals
            sqrt_term = np.sqrt(raw**2 + eps)
            
            # d(smooth_res)/d(val_i) = -0.5 * (1 - raw/sqrt_term)
            # d(smooth_res)/d(t) = 0.5 * (1 + raw/sqrt_term)
            
            d_res_d_val = -0.5 * (1.0 - raw / sqrt_term)
            d_res_dt = 0.5 * (1.0 + raw / sqrt_term)
            
            # Build Jacobian: (num_constraints, 2*N + 1)
            num_constraints = len(all_vals)
            J = np.zeros((num_constraints, 2*N + 1))
            
            # Pair constraints
            d_pair = dists[i_idx, j_idx]
            mask_nonzero = d_pair > 1e-10
            if np.any(mask_nonzero):
                # d(pair_val)/d(p_i) = (p_i - p_j) / (2 * d_pair)
                # d(pair_val)/d(p_j) = (p_j - p_i) / (2 * d_pair)
                diff_m = p[i_idx[mask_nonzero]] - p[j_idx[mask_nonzero]]
                d_pval_dp_i = diff_m / (2.0 * d_pair[mask_nonzero, np.newaxis])
                d_pval_dp_j = -d_pval_dp_i
                
                w = d_res_d_val[:num_pairs][mask_nonzero]
                for idx in range(len(mask_nonzero.nonzero()[0])):
                    m = mask_nonzero.nonzero()[0][idx]
                    row = m
                    J[row, 2*i_idx[m]:2*i_idx[m]+2] += w[idx] * d_pval_dp_i[idx]
                    J[row, 2*j_idx[m]:2*j_idx[m]+2] += w[idx] * d_pval_dp_j[idx]
            
            # Boundary constraints
            rad_safe = np.where(rad > 1e-10, rad, 1.0)
            d_bval_dp = -p / rad_safe[:, np.newaxis]
            w_bound = d_res_d_val[num_pairs:]
            J[num_pairs:, :2*N] += w_bound[:, np.newaxis] * d_bval_dp
            
            # d/dt column
            J[:, 2*N] = d_res_dt
            
            return J
        
        try:
            x0 = np.concatenate([pts.flatten(), [t]])
            
            result = least_squares(
                residuals,
                x0,
                jac=jacobian,
                method='trf',
                max_nfev=300,
                xtol=1e-14,
                ftol=1e-14,
                gtol=1e-14,
                bounds=([-1.5]*2*N + [0.0], [1.5]*2*N + [1.0])
            )
            
            new_pts = result.x[:2*N].reshape(N, 2)
            new_t = result.x[2*N]
            
            # Project inside circle
            rad = np.linalg.norm(new_pts, axis=1)
            mask_out = rad > 1.0
            if np.any(mask_out):
                new_pts[mask_out] /= rad[mask_out, np.newaxis]
            
            new_score = compute_score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                
        except Exception:
            pass
        
        return best_pts, best_score
    
    def active_set_newton(pts, time_budget):
        """
        Active-set Newton refinement on binding constraints.
        Identify the most binding constraints and solve a small linear system
        to move points to equalize these constraints.
        """
        if time.time() >= start_time + time_budget:
            return pts, compute_score(pts)
        
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        max_iters = 50
        tol = 1e-12
        
        for iteration in range(max_iters):
            if time.time() >= start_time + time_budget:
                break
            
            # Compute all distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            rad = np.linalg.norm(pts, axis=1)
            bound_dists = 1.0 - rad
            
            pair_vals = dists[i_idx, j_idx] / 2.0
            all_vals = np.concatenate([pair_vals, bound_dists])
            current_min = np.min(all_vals)
            
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
            
            # Find binding constraints (within tolerance of minimum)
            binding_threshold = current_min + 1e-6
            binding_mask = all_vals <= binding_threshold
            num_binding = np.sum(binding_mask)
            
            if num_binding == 0:
                break
            
            # Select up to 30 most binding constraints
            if num_binding > 30:
                binding_indices = np.argsort(all_vals)[:30]
            else:
                binding_indices = np.where(binding_mask)[0]
            
            # Build linear system: A * delta_x = b
            # For each binding constraint, we want to increase its value
            # The gradient of each constraint gives us the direction
            num_constraints = len(binding_indices)
            A = np.zeros((num_constraints, 2*N))
            
            for idx, ci in enumerate(binding_indices):
                if ci < num_pairs:
                    # Pair constraint
                    i, j = i_idx[ci], j_idx[ci]
                    d = dists[i, j]
                    if d < 1e-10:
                        continue
                    vec = (pts[i] - pts[j]) / d
                    A[idx, 2*i:2*i+2] = vec
                    A[idx, 2*j:2*j+2] = -vec
                else:
                    # Boundary constraint
                    k = ci - num_pairs
                    r = rad[k]
                    if r < 1e-10:
                        continue
                    vec = -pts[k] / r
                    A[idx, 2*k:2*k+2] = vec
            
            # Target: increase each binding constraint by a small amount
            # Use least squares to find delta_x
            try:
                # Solve A * delta_x = 1 (increase all constraints by 1)
                delta_x, residuals, rank, sv = np.linalg.lstsq(A, np.ones(num_constraints), rcond=None)
            except:
                break
            
            # Scale step
            step_size = 1e-4
            pts += step_size * delta_x.reshape(N, 2)
            
            # Project inside circle
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            # Check convergence
            new_score = compute_score(pts)
            if abs(new_score - current_min) < tol:
                break
        
        return best_pts, best_score
    
    def anneal_restart(pts, time_budget):
        """
        Anneal restart: perturb the best solution and re-optimize.
        """
        if time.time() >= start_time + time_budget:
            return pts, compute_score(pts)
        
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        num_restarts = 3
        time_per_restart = time_budget / num_restarts
        
        for r in range(num_restarts):
            if time.time() >= start_time + time_budget:
                break
            
            # Perturb
            noise_scale = 0.005 * (1 - r / num_restarts)
            pts_perturbed = pts + np.random.normal(0, noise_scale, size=pts.shape)
            rad_new = np.linalg.norm(pts_perturbed, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts_perturbed[mask_out] /= rad_new[mask_out, np.newaxis]
            
            # Re-optimize
            pts_opt, score = optimize(pts_perturbed,
                                       alpha_init=0.003,
                                       k_init=100.0,
                                       k_end=500.0,
                                       cooling=0.9995,
                                       time_budget=time_per_restart,
                                       stall_threshold=100,
                                       noise_scale_base=0.002)
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
        
        return best_pts, best_score
    
    def sharp_soft_min_polish(pts, time_budget):
        """
        Final polish with very sharp soft-min (k=1000+) and tiny steps.
        Focus on the most binding constraints.
        """
        if time.time() >= start_time + time_budget:
            return pts, compute_score(pts)
        
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        alpha = 0.001
        k = 500.0
        stall_count = 0
        end_time = start_time + time_budget
        
        while time.time() < end_time:
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            rad = np.linalg.norm(pts, axis=1)
            bound_dists = 1.0 - rad
            
            pair_vals = dists[i_idx, j_idx] / 2.0
            all_vals = np.concatenate([pair_vals, bound_dists])
            current_min = np.min(all_vals)
            
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            weights = soft_min_weights(all_vals, k, current_min)
            grads = compute_grads(pts, weights, dists, rad, bound_dists)
            
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1e-10:
                grads /= grad_norm
            
            pts += alpha * grads
            
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            alpha *= 0.995
            k *= 1.01
            
            if stall_count > 300:
                pts += np.random.normal(0, 0.001, size=pts.shape)
                rad_new = np.linalg.norm(pts, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    pts[mask_out] /= rad_new[mask_out, np.newaxis]
                stall_count = 0
                alpha = max(alpha, 0.0005)
                k = max(k, 1000.0)
        
        return best_pts, best_score
    
    def hex_lattice_init(spacing):
        """Generate hexagonal lattice points inside unit circle."""
        y_step = spacing * np.sqrt(3) / 2
        max_j = int(1.0 / y_step) + 2
        max_i = int(1.0 / spacing) + 2
        
        pts = []
        for j in range(-max_j, max_j + 1):
            y = j * y_step
            offset = 0 if j % 2 == 0 else spacing / 2
            for i in range(-max_i, max_i + 1):
                x = i * spacing + offset
                if np.hypot(x, y) < 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) > N:
            dists = np.linalg.norm(pts, axis=1)
            idx = np.argsort(dists)[:N]
            pts = pts[idx]
        elif len(pts) < N:
            while len(pts) < N:
                p = np.random.uniform(-0.95, 0.95, 2)
                if np.linalg.norm(p) < 0.95:
                    pts = np.vstack([pts, p])
            pts = pts[:N]
        
        return pts
    
    def random_init():
        """Random initialization with rejection sampling."""
        pts = []
        while len(pts) < N:
            p = np.random.uniform(-0.98, 0.98, 2)
            if np.linalg.norm(p) < 0.98:
                pts.append(p)
        return np.array(pts)
    
    # Generate multiple initializations
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Try hexagonal lattice with various spacings
    base_spacing = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    spacing_multipliers = [0.92, 0.95, 0.98, 1.0, 1.02, 1.05]
    
    for mult in spacing_multipliers:
        if time.time() - start_time > time_limit * 0.55:
            break
        
        spacing = base_spacing * mult
        pts = hex_lattice_init(spacing)
        
        time_budget = min(8.0, time_limit * 0.08)
        pts_opt, score = optimize(pts,
                                   alpha_init=0.015,
                                   k_init=3.0,
                                   k_end=200.0,
                                   cooling=0.9998,
                                   time_budget=time_budget,
                                   stall_threshold=100,
                                   noise_scale_base=0.008)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
    
    # Phase 2: Try random initializations
    if time.time() - start_time < time_limit * 0.65:
        for _ in range(3):
            if time.time() - start_time > time_limit * 0.65:
                break
            
            pts = random_init()
            time_budget = min(6.0, time_limit * 0.05)
            pts_opt, score = optimize(pts,
                                       alpha_init=0.01,
                                       k_init=2.0,
                                       k_end=150.0,
                                       cooling=0.9997,
                                       time_budget=time_budget,
                                       stall_threshold=80,
                                       noise_scale_base=0.01)
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
    
    # Phase 3: Polish with gradient-based optimization (sharper)
    if best_pts is not None and time.time() - start_time < time_limit * 0.75:
        remaining_time = time_limit * 0.80 - time.time()
        if remaining_time > 3.0:
            pts_polished, score_polished = optimize(best_pts,
                                                     alpha_init=0.003,
                                                     k_init=200.0,
                                                     k_end=1000.0,
                                                     cooling=0.999,
                                                     time_budget=min(remaining_time, 10.0),
                                                     stall_threshold=200,
                                                     noise_scale_base=0.002)
            
            if score_polished > best_score:
                best_score = score_polished
                best_pts = pts_polished
    
    # Phase 4: Active-set Newton refinement
    if best_pts is not None and time.time() - start_time < time_limit * 0.82:
        remaining_time = time_limit * 0.85 - time.time()
        if remaining_time > 2.0:
            pts_newton, score_newton = active_set_newton(best_pts, min(remaining_time, 5.0))
            
            if score_newton > best_score:
                best_score = score_newton
                best_pts = pts_newton
    
    # Phase 5: Anneal restart from perturbed best solution
    if best_pts is not None and time.time() - start_time < time_limit * 0.88:
        remaining_time = time_limit * 0.92 - time.time()
        if remaining_time > 3.0:
            pts_anneal, score_anneal = anneal_restart(best_pts, min(remaining_time, 10.0))
            
            if score_anneal > best_score:
                best_score = score_anneal
                best_pts = pts_anneal
    
    # Phase 6: Epigraph polish
    if best_pts is not None and time.time() - start_time < time_limit * 0.93:
        remaining_time = time_limit * 0.96 - time.time()
        if remaining_time > 3.0:
            pts_ep, score_ep = epigraph_polish(best_pts, min(remaining_time, 8.0))
            
            if score_ep > best_score:
                best_score = score_ep
                best_pts = pts_ep
    
    # Phase 7: Sharp soft-min final polish
    if best_pts is not None and time.time() - start_time < time_limit * 0.97:
        remaining_time = time_limit * 0.99 - time.time()
        if remaining_time > 2.0:
            pts_sharp, score_sharp = sharp_soft_min_polish(best_pts, min(remaining_time, 5.0))
            
            if score_sharp > best_score:
                best_score = score_sharp
                best_pts = pts_sharp
    
    # Final verification
    if best_pts is None:
        best_pts = hex_lattice_init(base_spacing)
    
    # Ensure all points are inside the circle
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    return best_pts
