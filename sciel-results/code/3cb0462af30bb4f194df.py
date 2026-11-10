# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    
    Maximizes D/M where D = H_{-1} (entropy of x-y distribution) and
    M = max(H_0, H_1, H_2) (max entropy of x, x+y, x+2y distributions).
    
    Uses product-of-marginals model P[y,x] = R[y]*C[x] for reduced search space.
    
    Returns:
        P: np.ndarray of shape (10, 10), non-negative, summing to 1.
    """
    n = 10
    deadline = time.time() + 580  # Safety margin before 630s kill
    
    x_idx = np.arange(n)
    y_idx = np.arange(n)
    
    # Precompute value matrices for each slope
    # P[y, x] corresponds to X=x, Y=y
    # Value for slope r: x + r*y
    diff_vals = x_idx[None, :] - y_idx[:, None]      # x - y
    x_vals = x_idx[None, :]                          # x
    sum_vals = x_idx[None, :] + y_idx[:, None]       # x + y
    sum2_vals = x_idx[None, :] + 2 * y_idx[:, None]  # x + 2*y
    
    def compute_entropies_from_product(R, C):
        """
        Compute H_{-1}, H_0, H_1, H_2 from product distribution P[y,x] = R[y]*C[x].
        R: shape (n,), C: shape (n,)
        """
        P = np.outer(R, C)
        
        def shannon_entropy_from_values(P, vals):
            unique_vals = np.unique(vals)
            probs = []
            for v in unique_vals:
                mask = vals == v
                p = P[mask].sum()
                if p > 1e-15:
                    probs.append(p)
            if len(probs) == 0:
                return 0.0
            return -np.sum(probs * np.log2(probs))
        
        H_neg1 = shannon_entropy_from_values(P, diff_vals)
        H_0 = shannon_entropy_from_values(P, x_vals)
        H_1 = shannon_entropy_from_values(P, sum_vals)
        H_2 = shannon_entropy_from_values(P, sum2_vals)
        
        return H_neg1, H_0, H_1, H_2
    
    def objective_from_logits(log_R, log_C):
        """
        Objective: minimize -D/M using logit parameters for R and C.
        log_R: shape (n,), log_C: shape (n,)
        """
        # Stable softmax for R and C
        R = np.exp(log_R - np.max(log_R))
        R = R / R.sum()
        C = np.exp(log_C - np.max(log_C))
        C = C / C.sum()
        
        H_neg1, H_0, H_1, H_2 = compute_entropies_from_product(R, C)
        M = max(H_0, H_1, H_2)
        
        if M < 1e-15:
            return 1e10
        
        ratio = H_neg1 / M
        return -ratio
    
    def full_objective(params):
        """Combine log_R and log_C into single vector for optimizer."""
        log_R = params[:n]
        log_C = params[n:]
        return objective_from_logits(log_R, log_C)
    
    # Generate diverse initializations for (R, C)
    rng = np.random.default_rng(12345)
    inits = []
    
    # 1. Uniform marginals
    inits.append(np.zeros(2 * n))
    
    # 2. R concentrated at y=0, C uniform
    for i in range(n):
        init = np.zeros(2 * n)
        init[i] = 5.0  # log_R[i] high
        inits.append(init)
    
    # 3. C concentrated at x=0, R uniform
    for i in range(n):
        init = np.zeros(2 * n)
        init[n + i] = 5.0  # log_C[i] high
        inits.append(init)
    
    # 4. Both concentrated
    for i in range(n):
        for j in range(n):
            init = np.zeros(2 * n)
            init[i] = 4.0
            init[n + j] = 4.0
            inits.append(init)
    
    # 5. Random initializations
    for _ in range(20):
        inits.append(rng.normal(0, 3, 2 * n))
    
    # 6. Structured: R peaked in middle, C peaked in middle
    mid = n // 2
    init = np.zeros(2 * n)
    init[mid] = 4.0
    init[n + mid] = 4.0
    inits.append(init)
    
    # 7. R uniform, C alternating high/low
    init = np.zeros(2 * n)
    for i in range(0, n, 2):
        init[n + i] = 3.0
    inits.append(init)
    
    # 8. R alternating, C uniform
    init = np.zeros(2 * n)
    for i in range(0, n, 2):
        init[i] = 3.0
    inits.append(init)
    
    best_P = None
    best_score = -np.inf
    
    # Also keep track of best full-distribution solution for fallback
    best_full_P = None
    best_full_score = -np.inf
    
    def optimize_from_init_product(init_params):
        """Run optimization from given initial logits for product model."""
        try:
            result = opt.minimize(
                full_objective,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.fun < 0:
                params_opt = result.x
                log_R = params_opt[:n]
                log_C = params_opt[n:]
                
                R = np.exp(log_R - np.max(log_R))
                R = R / R.sum()
                C = np.exp(log_C - np.max(log_C))
                C = C / C.sum()
                
                P = np.outer(R, C)
                H_neg1, H_0, H_1, H_2 = compute_entropies_from_product(R, C)
                M = max(H_0, H_1, H_2)
                if M > 1e-15:
                    score = H_neg1 / M
                    return P, score
        except Exception:
            pass
        return None, -np.inf
    
    # Optimize over product model
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
        
        P_opt, score = optimize_from_init_product(init)
        if P_opt is not None and score > best_score:
            best_score = score
            best_P = P_opt
    
    # Also try a few full-distribution optimizations as fallback
    def objective_full(params):
        """Objective for full 100-dim distribution."""
        params_clipped = np.clip(params, -50, 50)
        exp_params = np.exp(params_clipped - np.max(params_clipped))
        P = exp_params / exp_params.sum()
        P = P.reshape(n, n)
        
        def shannon_entropy_from_values(P, vals):
            unique_vals = np.unique(vals)
            probs = []
            for v in unique_vals:
                mask = vals == v
                p = P[mask].sum()
                if p > 1e-15:
                    probs.append(p)
            if len(probs) == 0:
                return 0.0
            return -np.sum(probs * np.log2(probs))
        
        H_neg1 = shannon_entropy_from_values(P, diff_vals)
        H_0 = shannon_entropy_from_values(P, x_vals)
        H_1 = shannon_entropy_from_values(P, sum_vals)
        H_2 = shannon_entropy_from_values(P, sum2_vals)
        M = max(H_0, H_1, H_2)
        
        if M < 1e-15:
            return 1e10
        return -(H_neg1 / M)
    
    # A few full-dim initializations
    full_inits = [
        np.zeros(n * n),  # uniform
        rng.normal(0, 2, n * n),
        rng.normal(0, 3, n * n),
    ]
    
    for init in full_inits:
        if time.time() > deadline:
            break
        try:
            result = opt.minimize(
                objective_full,
                init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            if result.fun < 0:
                params_opt = result.x
                exp_params = np.exp(np.clip(params_opt, -50, 50) - np.max(np.clip(params_opt, -50, 50)))
                P_opt = (exp_params / exp_params.sum()).reshape(n, n)
                
                def shannon_entropy_from_values(P, vals):
                    unique_vals = np.unique(vals)
                    probs = []
                    for v in unique_vals:
                        mask = vals == v
                        p = P[mask].sum()
                        if p > 1e-15:
                            probs.append(p)
                    if len(probs) == 0:
                        return 0.0
                    return -np.sum(probs * np.log2(probs))
                
                H_neg1 = shannon_entropy_from_values(P_opt, diff_vals)
                H_0 = shannon_entropy_from_values(P_opt, x_vals)
                H_1 = shannon_entropy_from_values(P_opt, sum_vals)
                H_2 = shannon_entropy_from_values(P_opt, sum2_vals)
                M = max(H_0, H_1, H_2)
                if M > 1e-15:
                    score = H_neg1 / M
                    if score > best_full_score:
                        best_full_score = score
                        best_full_P = P_opt
        except Exception:
            pass
    
    # Choose best between product model and full model
    if best_full_P is not None and best_full_score > best_score:
        best_P = best_full_P
        best_score = best_full_score
    
    # Fallback to uniform if no good result
    if best_P is None:
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    # Clip and normalize
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total > 1e-15:
        best_P = best_P / total
    else:
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    return best_P

# EVOLVE-BLOCK-END
