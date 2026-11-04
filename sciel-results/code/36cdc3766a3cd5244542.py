import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # Precompute pair indices for efficiency
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    def compute_score(pts):
        """Compute the packing radius score efficiently."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = np.sqrt(sq_dists[i_idx, j_idx]) / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        return np.min(all_vals)
    
    def _compute_vals_and_grads(pts, k):
        """
        Compute all constraint values, soft-min weights, and gradient in one pass.
        Returns: (all_vals, weights, grads)
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        d_pair_sq = sq_dists[i_idx, j_idx]
        d_pair = np.sqrt(d_pair_sq)
        
        pair_vals = d_pair / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        current_min = np.min(all_vals)
        
        # Compute soft-min weights
        shifted = all_vals - current_min
        # Only consider constraints within a reasonable range for numerical stability
        threshold = 40.0 / max(k, 1.0)
        mask_close = shifted < threshold
        if not np.any(mask_close):
            mask_close = np.ones_like(shifted, dtype=bool)
        
        close_shifted = shifted[mask_close]
        exp_vals = np.exp(-k * close_shifted)
        sum_exp = np.sum(exp_vals)
        
        weights = np.zeros_like(all_vals)
        weights[mask_close] = exp_vals / sum_exp
        
        # Compute gradients for maximizing soft_min (we'll negate for minimization)
        grads = np.zeros((N, 2))
        
        # Pair contributions
        pair_weights = weights[:num_pairs]
        mask_nonzero = d_pair > 1e-10
        if np.any(mask_nonzero):
            idx_m = mask_nonzero
            vecs = (diff[i_idx[idx_m], j_idx[idx_m]] / 
                    d_pair[idx_m, np.newaxis]) * pair_weights[idx_m, np.newaxis]
            
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        # Boundary contributions: d(bound_vals)/d(pts) = -pts/rad
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-10, rad, 1.0)
        bound_grads = -(pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        return all_vals, weights, grads, current_min
    
    def soft_min_objective(x, k):
        """
        Smooth surrogate: -soft_min of all constraint values.
        We minimize this to maximize the minimum constraint.
        Uses log-sum-exp for numerical stability.
        """
        pts = x.reshape(N, 2)
        all_vals, weights, grads, current_min = _compute_vals_and_grads(pts, k)
        
        # Soft min value
        shifted = all_vals - current_min
        threshold = 40.0 / max(k, 1.0)
        mask_close = shifted < threshold
        if not np.any(mask_close):
            mask_close = np.ones_like(shifted, dtype=bool)
        
        close_shifted = shifted[mask_close]
        log_sum = np.log(np.sum(np.exp(-k * close_shifted)))
        soft_min_val = current_min - log_sum / k
        
        return -soft_min_val
    
    def soft_min_grad(x, k):
        """
        Gradient of the soft-min objective w.r.t. point positions.
        We minimize -soft_min, so gradient is -d(soft_min)/d(pts).
        """
        pts = x.reshape(N, 2)
        all_vals, weights, grads, current_min = _compute_vals_and_grads(pts, k)
        
        # grads from _compute_vals_and_grads is d(soft_min)/d(pts)
        # We want d(-soft_min)/d(pts) = -grads
        return (-grads).flatten()
    
    def optimize(pts, alpha_init, k_init, k_end, cooling, time_budget, 
                 stall_threshold=150, noise_scale_base=0.01, polish=False):
        """
        Main optimization loop using soft-min gradient ascent.
        Returns best points found.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        alpha = alpha_init
        k = k_init
        stall_count = 0
        iter_count = 0
        end_time = start_time + time_budget
        
        if polish:
            stall_threshold = 300
            cooling = 0.9995
        
        while time.time() < end_time:
            # Compute values, weights, and gradients in one efficient pass
            all_vals, weights, grads, current_min = _compute_vals_and_grads(pts, k)
            
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            # Normalize gradient for stable updates
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1e-10:
                grads = grads / grad_norm * min(grad_norm, 1.0)
            
            # Adaptive step size
            if polish:
                step_alpha = alpha * 0.5
            else:
                step_alpha = alpha
            
            # Update (gradient ascent on soft_min = gradient descent on -soft_min)
            pts += step_alpha * grads
            
            # Project back inside circle
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            # Cool down
            alpha *= cooling
            
            # Increase k
            progress = min(1.0, (time.time() - start_time) / time_budget)
            if polish:
                k = k_init + (k_end - k_init) * (progress ** 2.0)
            else:
                k = k_init + (k_end - k_init) * (progress ** 1.5)
            
            # Random restart on stall
            if stall_count > stall_threshold:
                noise_scale = noise_scale_base * (1.0 + alpha * 10)
                if polish:
                    noise_scale *= 0.3
                pts += np.random.normal(0, noise_scale, size=pts.shape)
                rad_new = np.linalg.norm(pts, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    pts[mask_out] /= rad_new[mask_out, np.newaxis]
                stall_count = 0
                alpha = max(alpha, 0.005 if not polish else 0.002)
            
            iter_count += 1
        
        return best_pts, best_score
    
    # Generate multiple initializations
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
    
    # Try multiple initializations and optimization settings
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Try hexagonal lattice with various spacings
    base_spacing = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    spacing_multipliers = [0.95, 0.97, 0.99, 1.0, 1.01, 1.03]
    
    time_per_init = min(14.0, time_limit * 0.17)
    
    for idx, mult in enumerate(spacing_multipliers):
        if time.time() - start_time > time_limit * 0.68:
            break
        
        spacing = base_spacing * mult
        pts = hex_lattice_init(spacing)
        
        if 0.97 <= mult <= 1.03:
            time_budget = time_per_init
        else:
            time_budget = time_per_init * 0.6
        
        pts_opt, score = optimize(pts, 
                                   alpha_init=0.012, 
                                   k_init=5.0, 
                                   k_end=200.0, 
                                   cooling=0.9997,
                                   time_budget=time_budget,
                                   stall_threshold=120,
                                   noise_scale_base=0.006)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
    
    # Phase 2: Try random initializations
    if time.time() - start_time < time_limit * 0.78:
        for _ in range(2):
            if time.time() - start_time > time_limit * 0.78:
                break
            
            pts = random_init()
            time_budget = min(11.0, time_limit * 0.11)
            pts_opt, score = optimize(pts,
                                       alpha_init=0.008,
                                       k_init=4.0,
                                       k_end=150.0,
                                       cooling=0.9996,
                                       time_budget=time_budget,
                                       stall_threshold=100,
                                       noise_scale_base=0.008)
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
    
    # Phase 3: Polish the best result using L-BFGS-B with soft-min objective
    if best_pts is not None and time.time() - start_time < time_limit * 0.82:
        remaining_time = time_limit * 0.97 - time.time()
        
        if remaining_time > 12.0:
            x0 = best_pts.flatten()
            
            # First polish: moderate k
            k1 = 50.0
            obj0 = soft_min_objective(x0, k1)
            result1 = minimize(
                soft_min_objective, x0,
                jac=soft_min_grad,
                args=(k1,),
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 4000}
            )
            
            if result1.fun < obj0:
                pts1 = result1.x.reshape(N, 2)
                rad1 = np.linalg.norm(pts1, axis=1)
                mask_out1 = rad1 > 1.0
                if np.any(mask_out1):
                    pts1[mask_out1] /= rad1[mask_out1, np.newaxis]
                score1 = compute_score(pts1)
                if score1 > best_score:
                    best_score = score1
                    best_pts = pts1
            
            # Second polish: higher k
            remaining_time2 = time_limit * 0.97 - time.time()
            if remaining_time2 > 6.0:
                k2 = 200.0
                x2 = best_pts.flatten()
                obj2 = soft_min_objective(x2, k2)
                result2 = minimize(
                    soft_min_objective, x2,
                    jac=soft_min_grad,
                    args=(k2,),
                    method='L-BFGS-B',
                    options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 2500}
                )
                
                if result2.fun < obj2:
                    pts2 = result2.x.reshape(N, 2)
                    rad2 = np.linalg.norm(pts2, axis=1)
                    mask_out2 = rad2 > 1.0
                    if np.any(mask_out2):
                        pts2[mask_out2] /= rad2[mask_out2, np.newaxis]
                    score2 = compute_score(pts2)
                    if score2 > best_score:
                        best_score = score2
                        best_pts = pts2
            
            # Third polish: very high k
            remaining_time3 = time_limit * 0.97 - time.time()
            if remaining_time3 > 3.0:
                k3 = 500.0
                x3 = best_pts.flatten()
                obj3 = soft_min_objective(x3, k3)
                result3 = minimize(
                    soft_min_objective, x3,
                    jac=soft_min_grad,
                    args=(k3,),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 1500}
                )
                
                if result3.fun < obj3:
                    pts3 = result3.x.reshape(N, 2)
                    rad3 = np.linalg.norm(pts3, axis=1)
                    mask_out3 = rad3 > 1.0
                    if np.any(mask_out3):
                        pts3[mask_out3] /= rad3[mask_out3, np.newaxis]
                    score3 = compute_score(pts3)
                    if score3 > best_score:
                        best_score = score3
                        best_pts = pts3
    
    # Final verification
    if best_pts is None:
        # Fallback: simple hexagonal lattice
        best_pts = hex_lattice_init(base_spacing)
    
    # Ensure all points are inside the circle
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    return best_pts
