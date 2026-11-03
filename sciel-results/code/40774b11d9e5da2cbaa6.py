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
    
    def soft_min_objective(pts, k, current_min):
        """
        Soft-min surrogate: -log(sum(exp(-k*(d_i - current_min)))) / k
        This is a smooth approximation of min(d_i).
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        d_pair = np.sqrt(sq_dists[i_idx, j_idx]) / 2.0
        all_dists = np.concatenate([d_pair, bound_dists])
        
        shifted = all_dists - current_min
        # Clip to avoid numerical issues
        shifted = np.clip(shifted, -100, 100)
        exp_vals = np.exp(-k * shifted)
        log_sum = np.log(np.sum(exp_vals))
        return -log_sum / k
    
    def soft_min_gradient(pts, k, current_min):
        """
        Gradient of the soft-min objective.
        d/d(pts) [-log(sum(exp(-k*(d_i - current_min)))) / k]
        = sum_j [ (k * exp(-k*(d_j - current_min)) / sum(exp(-k*(d_i - current_min)))) * d(d_j)/d(pts) ] / k
        = sum_j [ w_j * d(d_j)/d(pts) ]
        where w_j = exp(-k*(d_j - current_min)) / sum(exp(-k*(d_i - current_min)))
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        d_pair = np.sqrt(sq_dists[i_idx, j_idx]) / 2.0
        all_dists = np.concatenate([d_pair, bound_dists])
        
        shifted = all_dists - current_min
        shifted = np.clip(shifted, -100, 100)
        exp_vals = np.exp(-k * shifted)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        grads = np.zeros_like(pts)
        
        # Pair contributions
        pair_weights = weights[:num_pairs]
        mask_nonzero = d_pair > 1e-10
        if np.any(mask_nonzero):
            idx_m = mask_nonzero
            vecs = (diff[i_idx[idx_m], j_idx[idx_m]] / 
                   d_pair[idx_m, np.newaxis]) * pair_weights[idx_m, np.newaxis]
            
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        # Boundary contributions
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-10, rad, 1.0)
        bound_grads = -(pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        return grads
    
    def optimize_soft_min(pts, alpha_init, k_init, k_end, cooling, time_budget, 
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
            # Compute distances
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
            
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            # Compute gradient of soft-min
            grads = soft_min_gradient(pts, k, current_min)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1.0:
                grads /= grad_norm
            
            # Adaptive step size
            if polish:
                step_alpha = alpha * 0.5
            else:
                step_alpha = alpha
            
            # Update
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
    
    def kkt_polish(pts, time_budget, max_active=200, tol=1e-12):
        """
        KKT-based active-set polish.
        
        At the optimum, the configuration is defined by a rigid network of touching
        circles. We identify the binding constraints (near-minimum distances) and
        solve the KKT linear system for the exact positions and radius.
        
        The KKT conditions for maximizing r subject to:
          |c_i - c_j|^2 >= (2r)^2 for all i < j  (pair constraints)
          |c_i|^2 + (2r)^2 <= 1 for all i  (boundary constraints)
        
        are:
          For active pair (i,j): (c_i - c_j) / |c_i - c_j| = lambda_ij * (e_i - e_j)
          For active boundary i: c_i / |c_i| = mu_i * e_i
          where e_i is the unit vector in direction i, and lambda, mu are multipliers.
        
        We solve the linear system:
          A * [delta_c; delta_r] = b
        where A is built from the active constraints.
        """
        end_time = start_time + time_budget
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        # Precompute pair indices
        i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
        num_pairs = len(i_idx)
        
        for iteration in range(50):
            if time.time() > end_time - 1.0:
                break
            
            # Compute all distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            
            rad = np.linalg.norm(pts, axis=1)
            bound_dists = 1.0 - rad
            
            d_pair = np.sqrt(sq_dists[i_idx, j_idx])
            pair_vals = d_pair / 2.0
            all_vals = np.concatenate([pair_vals, bound_dists])
            current_r = np.min(all_vals)
            
            if current_r > best_score:
                best_score = current_r
                best_pts = pts.copy()
            
            # Identify active constraints: those within tolerance of the minimum
            # Use a relative tolerance to select the "tightest" constraints
            tol_active = max(1e-8, current_r * 1e-6)
            
            # Active pairs: distance/2 is close to current_r
            active_pairs = np.where(pair_vals <= current_r + tol_active)[0]
            # Active boundaries: 1 - |c_i| is close to current_r
            active_bounds = np.where(bound_dists <= current_r + tol_active)[0]
            
            # Limit active set size for numerical stability
            if len(active_pairs) > max_active:
                # Keep the tightest ones
                pair_vals_sorted = np.argsort(pair_vals)
                active_pairs = pair_vals_sorted[:max_active]
            if len(active_bounds) > 100:
                bound_dists_sorted = np.argsort(bound_dists)
                active_bounds = bound_dists_sorted[:100]
            
            if len(active_pairs) == 0 and len(active_bounds) == 0:
                break
            
            # Build KKT system
            # Variables: delta_c (2N) + delta_r (1) = 2N+1 unknowns
            # For each active pair (i,j):
            #   (c_i - c_j)^T * (delta_c_i - delta_c_j) / d_ij = 2 * delta_r
            # For each active boundary i:
            #   c_i^T * delta_c_i / |c_i| = 2 * delta_r
            # Plus: we want to maximize r, so we set delta_r = 1 (direction of increase)
            # Actually, we solve for the direction that increases r the most.
            
            # Better formulation: we want to find delta such that
            # all active constraints remain satisfied to first order,
            # and we maximize the increase in r.
            
            # Let's set up: find delta_c, delta_r such that:
            #   For active pair (i,j): (c_i - c_j)^T (delta_c_i - delta_c_j) / d_ij - 2*delta_r = 0
            #   For active boundary i: c_i^T delta_c_i / |c_i| - 2*delta_r = 0
            #   And we maximize delta_r (or equivalently, set delta_r = 1 and minimize ||delta_c||)
            
            # Actually, the standard approach: solve for the direction that increases r
            # subject to the active constraints being tight.
            
            # Let's use a simpler approach: solve the least-squares problem
            # to find the best local improvement direction.
            
            # Build constraint matrix A (m x (2N+1)) and rhs b (m)
            m = len(active_pairs) + len(active_bounds)
            if m == 0:
                break
            
            A = np.zeros((m, 2*N + 1))
            b = np.zeros(m)
            
            # Pair constraints
            for idx, pair_idx in enumerate(active_pairs):
                i = i_idx[pair_idx]
                j = j_idx[pair_idx]
                d = d_pair[pair_idx]
                if d < 1e-10:
                    continue
                vec = diff[i, j] / d  # unit vector from j to i
                A[idx, 2*i] = vec[0]
                A[idx, 2*i+1] = vec[1]
                A[idx, 2*j] = -vec[0]
                A[idx, 2*j+1] = -vec[1]
                A[idx, 2*N] = -2.0
                b[idx] = 0.0
            
            # Boundary constraints
            for idx, bound_idx in enumerate(active_bounds):
                i = bound_idx
                r_i = rad[i]
                if r_i < 1e-10:
                    continue
                vec = pts[i] / r_i  # unit vector from origin to i
                A[len(active_pairs) + idx, 2*i] = vec[0]
                A[len(active_pairs) + idx, 2*i+1] = vec[1]
                A[len(active_pairs) + idx, 2*N] = -2.0
                b[len(active_pairs) + idx] = 0.0
            
            # Solve least squares: minimize ||A * x - b||^2
            # x = [delta_c; delta_r]
            try:
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=tol)
            except np.linalg.LinAlgError:
                break
            
            if x is None or not np.all(np.isfinite(x)):
                break
            
            delta_c = x[:2*N].reshape(N, 2)
            delta_r = x[2*N]
            
            # Check if this direction increases r
            if delta_r <= 0:
                # Try the opposite direction
                delta_c = -delta_c
                delta_r = -delta_r
                if delta_r <= 0:
                    break
            
            # Line search: find the largest step that maintains feasibility
            max_step = 0.01  # Initial guess
            for _ in range(30):
                test_pts = pts + max_step * delta_c
                test_r = compute_score(test_pts)
                if test_r >= current_r - 1e-12:
                    break
                max_step *= 0.5
            
            if max_step < 1e-15:
                break
            
            pts = pts + max_step * delta_c
            
            # Project inside circle
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
        
        return best_pts, best_score
    
    def polish_with_scipy(pts, time_budget):
        """
        Use scipy minimize to polish the packing.
        We minimize the negative soft-min with a high k value.
        """
        end_time = start_time + time_budget
        current_best = compute_score(pts)
        
        # Use a high k for near-exact min
        k_final = 500.0
        
        def objective(x):
            pts_x = x.reshape(N, 2)
            return soft_min_objective(pts_x, k_final, current_best)
        
        def gradient(x):
            pts_x = x.reshape(N, 2)
            return soft_min_gradient(pts_x, k_final, current_best).flatten()
        
        x0 = pts.flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 30}
            )
            pts_result = result.x.reshape(N, 2)
            
            # Project back inside
            rad = np.linalg.norm(pts_result, axis=1)
            mask_out = rad > 1.0
            if np.any(mask_out):
                pts_result[mask_out] /= rad[mask_out, np.newaxis]
            
            score_result = compute_score(pts_result)
            return pts_result, score_result
        except Exception:
            return pts, current_best
    
    # Generate multiple initializations
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Try hexagonal lattice with various spacings
    base_spacing = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    spacing_multipliers = [0.95, 0.97, 0.99, 1.0, 1.01, 1.03]
    
    time_per_init = min(15.0, time_limit * 0.18)
    
    for idx, mult in enumerate(spacing_multipliers):
        if time.time() - start_time > time_limit * 0.70:
            break
        
        spacing = base_spacing * mult
        pts = hex_lattice_init(spacing)
        
        if 0.97 <= mult <= 1.03:
            time_budget = time_per_init
        else:
            time_budget = time_per_init * 0.6
        
        pts_opt, score = optimize_soft_min(pts, 
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
    if time.time() - start_time < time_limit * 0.80:
        for _ in range(2):
            if time.time() - start_time > time_limit * 0.80:
                break
            
            pts = random_init()
            time_budget = min(12.0, time_limit * 0.12)
            pts_opt, score = optimize_soft_min(pts,
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
    
    # Phase 3: Polish the best result
    if best_pts is not None and time.time() - start_time < time_limit * 0.90:
        remaining_time = time_limit * 0.97 - time.time()
        if remaining_time > 5.0:
            # First polish pass with soft-min gradient ascent
            pts_polished, score_polished = optimize_soft_min(best_pts,
                                                             alpha_init=0.005,
                                                             k_init=100.0,
                                                             k_end=300.0,
                                                             cooling=0.9995,
                                                             time_budget=min(remaining_time * 0.25, 6.0),
                                                             stall_threshold=150,
                                                             noise_scale_base=0.003,
                                                             polish=True)
            
            if score_polished > best_score:
                best_score = score_polished
                best_pts = pts_polished
            
            # Second: Use KKT active-set polish
            remaining_time2 = time_limit * 0.97 - time.time()
            if remaining_time2 > 3.0:
                pts_kkt, score_kkt = kkt_polish(best_pts, min(remaining_time2 * 0.4, 10.0))
                
                if score_kkt > best_score:
                    best_score = score_kkt
                    best_pts = pts_kkt
            
            # Third: Use scipy for high-precision polish
            remaining_time3 = time_limit * 0.97 - time.time()
            if remaining_time3 > 3.0:
                pts_scipy, score_scipy = polish_with_scipy(best_pts, min(remaining_time3 * 0.5, 12.0))
                
                if score_scipy > best_score:
                    best_score = score_scipy
                    best_pts = pts_scipy
            
            # Fourth: Another KKT pass if time allows
            remaining_time4 = time_limit * 0.97 - time.time()
            if remaining_time4 > 2.0:
                pts_kkt2, score_kkt2 = kkt_polish(best_pts, min(remaining_time4 * 0.7, 8.0))
                
                if score_kkt2 > best_score:
                    best_score = score_kkt2
                    best_pts = pts_kkt2
    
    # Final verification
    if best_pts is None:
        best_pts = hex_lattice_init(base_spacing)
    
    # Ensure all points are inside the circle
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    # Final high-precision polish if time allows
    final_remaining = time_limit * 0.98 - time.time()
    if final_remaining > 2.0:
        # Try KKT first, then scipy
        pts_final, score_final = kkt_polish(best_pts, final_remaining * 0.5)
        if score_final > best_score:
            best_score = score_final
            best_pts = pts_final
        
        final_remaining2 = time_limit * 0.98 - time.time()
        if final_remaining2 > 1.0:
            pts_final2, score_final2 = polish_with_scipy(best_pts, final_remaining2)
            if score_final2 > best_score:
                best_score = score_final2
                best_pts = pts_final2
    
    # Final safety check
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    return best_pts
