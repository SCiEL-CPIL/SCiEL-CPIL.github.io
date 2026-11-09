# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 600s kill

    n = 10
    ys = np.arange(n)
    xs = np.arange(n)
    X, Y = np.meshgrid(xs, ys)  # X[y,x] = x, Y[y,x] = y

    # Precompute the index arrays for each projection
    XY_diff = (X - Y + (n - 1)).flatten()  # 0 to 2*(n-1)
    X_flat = X.flatten()  # 0 to n-1
    XY_sum = (X + Y).flatten()  # 0 to 2*(n-1)
    X2Y = (X + 2 * Y).flatten()  # 0 to 3*(n-1)
    
    # Precompute mapping: for each cell, which bin it falls into for each projection
    # This allows efficient gradient computation
    n_bins_diff = 2 * n - 1  # 19
    n_bins_x = n  # 10
    n_bins_sum = 2 * n - 1  # 19
    n_bins_x2y = 3 * n - 1  # 29

    def compute_ratio(P):
        """Compute D/M ratio for a given distribution P (vectorized)."""
        P = np.maximum(P, 0)
        total = P.sum()
        if total <= 1e-9:
            return 0.0
        P = P / total
        P_flat = P.flatten()

        # H(X-Y)
        XY_dist = np.bincount(XY_diff, weights=P_flat, minlength=n_bins_diff)
        XY_nonzero = XY_dist[XY_dist > 1e-15]
        H_XY = -np.sum(XY_nonzero * np.log2(XY_nonzero)) if len(XY_nonzero) > 0 else 0.0

        # H(X)
        X_dist = np.bincount(X_flat, weights=P_flat, minlength=n_bins_x)
        X_nonzero = X_dist[X_dist > 1e-15]
        H_X = -np.sum(X_nonzero * np.log2(X_nonzero)) if len(X_nonzero) > 0 else 0.0

        # H(X+Y)
        XY_sum_dist = np.bincount(XY_sum, weights=P_flat, minlength=n_bins_sum)
        XY_sum_nonzero = XY_sum_dist[XY_sum_dist > 1e-15]
        H_XY_sum = -np.sum(XY_sum_nonzero * np.log2(XY_sum_nonzero)) if len(XY_sum_nonzero) > 0 else 0.0

        # H(X+2Y)
        X2Y_dist = np.bincount(X2Y, weights=P_flat, minlength=n_bins_x2y)
        X2Y_nonzero = X2Y_dist[X2Y_dist > 1e-15]
        H_X2Y = -np.sum(X2Y_nonzero * np.log2(X2Y_nonzero)) if len(X2Y_nonzero) > 0 else 0.0

        M = max(H_X, H_XY_sum, H_X2Y)
        if M <= 1e-9:
            return 0.0
        return H_XY / M

    def compute_ratio_with_grad(params):
        """
        Compute -D/M and its gradient w.r.t. log-parameters.
        P = exp(params) / sum(exp(params))
        """
        log_P = params.reshape(n, n)
        # Softmax for numerical stability
        log_P_shifted = log_P - np.max(log_P)
        P_unnorm = np.exp(log_P_shifted)
        total = P_unnorm.sum()
        if total <= 1e-9:
            return 1e10, np.zeros_like(params)
        P = P_unnorm / total
        P_flat = P.flatten()

        # Compute entropies and their gradients
        # For H = -sum(p_i log2(p_i)), dH/dP_j = -log2(p_bin) - 1/ln(2) if P_j contributes to bin
        # where p_bin = sum of P_j over cells in that bin
        
        # H(X-Y)
        XY_dist = np.bincount(XY_diff, weights=P_flat, minlength=n_bins_diff)
        XY_nonzero_mask = XY_dist > 1e-15
        H_XY = 0.0
        if XY_nonzero_mask.any():
            XY_nonzero = XY_dist[XY_nonzero_mask]
            H_XY = -np.sum(XY_nonzero * np.log2(XY_nonzero))
        
        # Gradient of H_XY w.r.t. P_flat
        # dH_XY/dP_j = -(log2(XY_dist[bin_j]) + 1/ln(2)) / XY_dist[bin_j] * XY_dist[bin_j]
        # = -(log2(XY_dist[bin_j]) + 1/ln(2))
        # Wait, let me redo: H = -sum_b p_b log2(p_b), p_b = sum_{j in bin_b} P_j
        # dH/dP_j = -d/dP_j [sum_b p_b log2(p_b)] = -sum_b [log2(p_b) * dp_b/dP_j + p_b * (1/(p_b ln2)) * dp_b/dP_j]
        # = -sum_b [log2(p_b) + 1/ln(2)] * dp_b/dP_j
        # dp_b/dP_j = 1 if j in bin_b, else 0
        # So dH/dP_j = -(log2(p_bin(j)) + 1/ln(2))
        
        if XY_nonzero_mask.any():
            XY_log2 = np.where(XY_nonzero_mask, np.log2(np.maximum(XY_dist, 1e-300)), 0.0)
            dH_XY_dP = -(XY_log2[XY_diff] + 1.0 / np.log(2))
            # Only for nonzero bins
            dH_XY_dP = np.where(XY_dist[XY_diff] > 1e-15, dH_XY_dP, 0.0)
        else:
            dH_XY_dP = np.zeros(n * n)

        # H(X)
        X_dist = np.bincount(X_flat, weights=P_flat, minlength=n_bins_x)
        X_nonzero_mask = X_dist > 1e-15
        H_X = 0.0
        if X_nonzero_mask.any():
            X_nonzero = X_dist[X_nonzero_mask]
            H_X = -np.sum(X_nonzero * np.log2(X_nonzero))
        
        if X_nonzero_mask.any():
            X_log2 = np.where(X_nonzero_mask, np.log2(np.maximum(X_dist, 1e-300)), 0.0)
            dH_X_dP = -(X_log2[X_flat] + 1.0 / np.log(2))
            dH_X_dP = np.where(X_dist[X_flat] > 1e-15, dH_X_dP, 0.0)
        else:
            dH_X_dP = np.zeros(n * n)

        # H(X+Y)
        XY_sum_dist = np.bincount(XY_sum, weights=P_flat, minlength=n_bins_sum)
        XY_sum_nonzero_mask = XY_sum_dist > 1e-15
        H_XY_sum = 0.0
        if XY_sum_nonzero_mask.any():
            XY_sum_nonzero = XY_sum_dist[XY_sum_nonzero_mask]
            H_XY_sum = -np.sum(XY_sum_nonzero * np.log2(XY_sum_nonzero))
        
        if XY_sum_nonzero_mask.any():
            XY_sum_log2 = np.where(XY_sum_nonzero_mask, np.log2(np.maximum(XY_sum_dist, 1e-300)), 0.0)
            dH_XY_sum_dP = -(XY_sum_log2[XY_sum] + 1.0 / np.log(2))
            dH_XY_sum_dP = np.where(XY_sum_dist[XY_sum] > 1e-15, dH_XY_sum_dP, 0.0)
        else:
            dH_XY_sum_dP = np.zeros(n * n)

        # H(X+2Y)
        X2Y_dist = np.bincount(X2Y, weights=P_flat, minlength=n_bins_x2y)
        X2Y_nonzero_mask = X2Y_dist > 1e-15
        H_X2Y = 0.0
        if X2Y_nonzero_mask.any():
            X2Y_nonzero = X2Y_dist[X2Y_nonzero_mask]
            H_X2Y = -np.sum(X2Y_nonzero * np.log2(X2Y_nonzero))
        
        if X2Y_nonzero_mask.any():
            X2Y_log2 = np.where(X2Y_nonzero_mask, np.log2(np.maximum(X2Y_dist, 1e-300)), 0.0)
            dH_X2Y_dP = -(X2Y_log2[X2Y] + 1.0 / np.log(2))
            dH_X2Y_dP = np.where(X2Y_dist[X2Y] > 1e-15, dH_X2Y_dP, 0.0)
        else:
            dH_X2Y_dP = np.zeros(n * n)

        M = max(H_X, H_XY_sum, H_X2Y)
        if M <= 1e-9:
            return 1e10, np.zeros_like(params)
        
        ratio = H_XY / M
        
        # Gradient of ratio w.r.t. P: d(ratio)/dP = (dH_XY/dP * M - H_XY * dM/dP) / M^2
        # dM/dP: M is max of three, so gradient is from the argmax
        # For simplicity, use subgradient from the max
        max_idx = np.argmax([H_X, H_XY_sum, H_X2Y])
        if max_idx == 0:
            dM_dP = dH_X_dP
        elif max_idx == 1:
            dM_dP = dH_XY_sum_dP
        else:
            dM_dP = dH_X2Y_dP
        
        d_ratio_dP = (dH_XY_dP * M - H_XY * dM_dP) / (M * M)
        
        # Chain rule: dP/dparams for softmax
        # P = softmax(params), dP_i/dparams_j = P_i * (delta_ij - P_j)
        # So d(f)/dparams_j = sum_i (df/dP_i) * P_i * (delta_ij - P_j)
        # = P_j * (df/dP_j - sum_i df/dP_i * P_i)
        
        P_flat_arr = P_flat
        sum_df_dP = np.sum(d_ratio_dP * P_flat_arr)
        d_ratio_dparams = P_flat_arr * (d_ratio_dP - sum_df_dP)
        
        # We're minimizing -ratio, so negate
        objective_val = -ratio
        grad = -d_ratio_dparams
        
        return objective_val, grad.reshape(n, n)

    def objective(params):
        """Negative of the ratio for minimization (exact max)."""
        val, _ = compute_ratio_with_grad(params)
        return val

    def objective_grad(params):
        """Negative of the ratio with gradient for L-BFGS-B."""
        val, grad = compute_ratio_with_grad(params)
        return val, grad.flatten()

    # Start with structured initializations
    np.random.seed(42)
    
    # Initial point 1: uniform on anti-diagonals (maximizes H(X-Y))
    P_best = np.zeros((n, n), dtype=np.float64)
    for y in range(n):
        for x in range(n):
            k = abs(x - y)
            P_best[y, x] = 1.0 / (19.0 * (n - k))
    P_best /= P_best.sum()
    best_ratio = compute_ratio(P_best)

    # Try L-BFGS-B from multiple diverse starting points
    starting_points = []
    
    # 1. Anti-diagonal uniform (current best)
    starting_points.append(P_best.copy())
    
    # 2. Concentrate on cells where x-y has high entropy but x, x+y, x+2y are concentrated
    for subset_size in [3, 4, 5, 6, 7]:
        for offset in range(n):
            P_candidate = np.zeros((n, n), dtype=np.float64)
            for y in range(n):
                for x in range(n):
                    d = (x - y + n) % (2*n - 1)
                    if d in range(offset, offset + subset_size) or d in range(2*n-1 - subset_size, 2*n-1):
                        P_candidate[y, x] = 1.0
            if P_candidate.sum() > 0:
                P_candidate /= P_candidate.sum()
                starting_points.append(P_candidate.copy())
    
    # 3. Random Dirichlet samples
    for _ in range(15):
        P_rand = np.random.dirichlet(np.ones(n*n))
        P_rand = P_rand.reshape(n, n)
        starting_points.append(P_rand)
    
    # 4. Structured: emphasize certain rows/columns
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        P_struct = np.zeros((n, n), dtype=np.float64)
        for y in range(n):
            for x in range(n):
                P_struct[y, x] = np.exp(-alpha * (x - y)**2 / (2*n))
        P_struct /= P_struct.sum()
        starting_points.append(P_struct)
    
    # 5. Sparse patterns
    for trial in range(10):
        P_sparse = np.zeros((n, n), dtype=np.float64)
        for _ in range(20):
            ry = np.random.randint(0, n)
            rx = np.random.randint(0, n)
            P_sparse[ry, rx] = 1.0
        P_sparse /= P_sparse.sum()
        starting_points.append(P_sparse)
    
    # 6. Additional structured initializations targeting the objective
    # Try to make H(X-Y) high while keeping H(X), H(X+Y), H(X+2Y) low
    # This means concentrating on a band where x-y varies but x, x+y, x+2y don't
    for band_width in [2, 3, 4]:
        for center in range(-5, 6):
            P_band = np.zeros((n, n), dtype=np.float64)
            for y in range(n):
                for x in range(n):
                    d = x - y
                    if abs(d - center) <= band_width:
                        P_band[y, x] = 1.0
            if P_band.sum() > 0:
                P_band /= P_band.sum()
                starting_points.append(P_band.copy())

    # Optimize from each starting point using L-BFGS-B with gradient
    for i, P_init in enumerate(starting_points):
        if time.time() > deadline - 30:
            break
        
        P_init_safe = np.maximum(P_init, 1e-12)
        params_init = np.log(P_init_safe.flatten())
        
        try:
            # Use L-BFGS-B with analytical gradient
            result = minimize(objective_grad, params_init, method='L-BFGS-B',
                             jac=True,
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000})
            
            P_opt = np.exp(result.x).reshape(n, n)
            P_opt = np.maximum(P_opt, 0)
            total = P_opt.sum()
            if total > 1e-9:
                P_opt /= total
                ratio = compute_ratio(P_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    P_best = P_opt.copy()
        except Exception:
            pass
    
    # Final polish: coordinate descent with binary search for optimal transfer
    P_work = P_best.copy()
    for iteration in range(20):
        if time.time() > deadline - 15:
            break
        
        improved = False
        for idx in range(n * n):
            if time.time() > deadline - 15:
                break
            old_val = P_work.flat[idx]
            if old_val < 1e-12:
                continue
            # Try moving mass from this cell to others
            for target_idx in range(n * n):
                if target_idx == idx:
                    continue
                # Binary search for optimal fraction to transfer
                lo, hi = 0.0, 1.0
                best_frac = 0.0
                best_trial_ratio = best_ratio
                for _ in range(10):
                    mid = (lo + hi) / 2.0
                    move_amt = old_val * mid
                    if move_amt < 1e-14:
                        continue
                    P_trial = P_work.copy()
                    P_trial.flat[idx] -= move_amt
                    P_trial.flat[target_idx] += move_amt
                    P_trial = np.maximum(P_trial, 0)
                    total = P_trial.sum()
                    if total < 1e-9:
                        continue
                    P_trial /= total
                    trial_ratio = compute_ratio(P_trial)
                    if trial_ratio > best_trial_ratio:
                        best_trial_ratio = trial_ratio
                        best_frac = mid
                        lo = mid
                    else:
                        hi = mid
                
                if best_frac > 0 and best_trial_ratio > best_ratio + 1e-10:
                    move_amt = old_val * best_frac
                    P_work.flat[idx] -= move_amt
                    P_work.flat[target_idx] += move_amt
                    P_work = np.maximum(P_work, 0)
                    total = P_work.sum()
                    P_work /= total
                    best_ratio = best_trial_ratio
                    improved = True
        
        if not improved:
            break
    
    P_best = P_work
    
    # Final high-precision polish: run L-BFGS-B one more time from the best point
    if time.time() < deadline - 30:
        P_best_safe = np.maximum(P_best, 1e-12)
        params_final = np.log(P_best_safe.flatten())
        try:
            result_final = minimize(objective_grad, params_final, method='L-BFGS-B',
                                   jac=True,
                                   options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 20000})
            P_final = np.exp(result_final.x).reshape(n, n)
            P_final = np.maximum(P_final, 0)
            total = P_final.sum()
            if total > 1e-9:
                P_final /= total
                ratio_final = compute_ratio(P_final)
                if ratio_final > best_ratio:
                    best_ratio = ratio_final
                    P_best = P_final.copy()
        except Exception:
            pass
    
    # Final normalization and clipping
    P_best = np.maximum(P_best, 0)
    P_best /= P_best.sum()

    return P_best

# EVOLVE-BLOCK-END
