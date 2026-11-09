import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    # Precompute bin indices for each slope
    # P is indexed [y, x] in the return.
    # ys, xs from np.indices((n, n)) gives ys[i,j]=i (y), xs[i,j]=j (x).
    ys, xs = np.indices((n, n))
    
    # Calculate values for each slope
    # Slope -1: x - y
    vals_m1 = xs - ys
    # Slope 0: x
    vals_0 = xs
    # Slope 1: x + y
    vals_1 = xs + ys
    # Slope 2: x + 2y
    vals_2 = xs + 2 * ys
    
    # Flatten for bincount
    bin_idx_m1 = (vals_m1 + 9).flatten()
    bin_idx_0 = vals_0.flatten()
    bin_idx_1 = vals_1.flatten()
    bin_idx_2 = vals_2.flatten()

    num_bins_m1 = 19
    num_bins_0 = 10
    num_bins_1 = 19
    num_bins_2 = 28

    def compute_ratio_from_logP(logP_flat, soft_max_beta=0.0):
        # Convert log-probabilities to probabilities
        # Subtract max for numerical stability
        z = logP_flat - np.max(logP_flat)
        P_flat = np.exp(z)
        sum_P = np.sum(P_flat)
        if sum_P < 1e-15:
            return 0.0
        P_flat = P_flat / sum_P
        
        # Compute bin counts (probabilities)
        counts_m1 = np.bincount(bin_idx_m1, weights=P_flat, minlength=num_bins_m1)
        counts_0 = np.bincount(bin_idx_0, weights=P_flat, minlength=num_bins_0)
        counts_1 = np.bincount(bin_idx_1, weights=P_flat, minlength=num_bins_1)
        counts_2 = np.bincount(bin_idx_2, weights=P_flat, minlength=num_bins_2)
        
        def entropy(counts):
            # Filter out zeros to avoid log(0)
            c = counts[counts > 1e-15]
            if len(c) == 0:
                return 0.0
            return -np.sum(c * np.log2(c))
            
        D = entropy(counts_m1)
        H0 = entropy(counts_0)
        H1 = entropy(counts_1)
        H2 = entropy(counts_2)
        
        # M = max(H0, H1, H2)
        # Use soft max for smooth optimization if beta > 0
        H_vals = np.array([H0, H1, H2])
        if soft_max_beta > 0:
            # Soft max approximation of max
            M = (1.0 / soft_max_beta) * np.log(np.sum(np.exp(soft_max_beta * H_vals)))
        else:
            M = np.max(H_vals)
            
        if M < 1e-9:
            return 0.0
            
        return D / M

    def objective(logP_flat, soft_max_beta=0.0):
        r = compute_ratio_from_logP(logP_flat, soft_max_beta)
        return -r

    start_time = time.time()
    time_limit = 620.0  # Safety margin for 630s limit
    
    best_score = 0.0
    best_logP = np.zeros(100)
    best_P = np.ones((10, 10)) / 100.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    init_points = []
    
    # 1. Uniform (all zeros in log space)
    init_points.append(np.zeros(100))
    
    # 2. Random Gaussian
    for _ in range(20):
        init_points.append(rng.normal(0, 1, 100))
        
    # 3. Sparse binary patterns (high weights on specific cells)
    # These can help escape local minima where continuous optimization gets stuck
    for _ in range(30):
        p = np.zeros(100)
        # Select random subset of cells
        idx = rng.choice(100, size=rng.integers(5, 20), replace=False)
        p[idx] = 8.0  # High log weight
        init_points.append(p)
        
    # 4. Diagonal patterns (x - y = const)
    for d in range(-9, 10):
        p = np.zeros(100)
        for y in range(10):
            x = y + d
            if 0 <= x < 10:
                p[y*10+x] = 8.0
        if np.sum(p) > 0:
            init_points.append(p)
            
    # 5. Anti-diagonal patterns (x + y = const)
    for s in range(0, 19):
        p = np.zeros(100)
        for y in range(10):
            x = s - y
            if 0 <= x < 10:
                p[y*10+x] = 8.0
        if np.sum(p) > 0:
            init_points.append(p)

    # 6. Structured 2D patterns
    # Checkerboard
    p = np.zeros(100)
    for y in range(10):
        for x in range(10):
            if (x+y) % 2 == 0:
                p[y*10+x] = 6.0
    init_points.append(p)
    
    # Phase 1: Broad search with soft max for smoothness
    # We use a moderate beta to smooth the max operation
    beta_phase1 = 50.0
    
    for i, init in enumerate(init_points):
        if time.time() - start_time > time_limit - 60:
            break
            
        try:
            # Use L-BFGS-B as it handles box constraints and is efficient for large N
            # We don't have explicit bounds on logP, but softmax makes it scale invariant.
            # We can fix the mean of logP to 0 to prevent drift, or just let it be.
            # L-BFGS-B works well here.
            res = minimize(lambda lp: objective(lp, beta_phase1), init, method='L-BFGS-B', 
                          options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-8})
            
            score = -res.fun
            if score > best_score:
                best_score = score
                best_logP = res.x.copy()
                # Convert to P for storage
                z = best_logP - np.max(best_logP)
                P_flat = np.exp(z)
                P_flat = P_flat / np.sum(P_flat)
                best_P = P_flat.reshape(10, 10)
        except Exception:
            pass

    # Phase 2: Refine with exact max (beta=0) using the best point from Phase 1
    if time.time() - start_time < time_limit - 30:
        try:
            # We might get stuck at kinks if we switch directly. 
            # Let's do a few iterations with decreasing beta.
            current_logP = best_logP.copy()
            betas = [20.0, 10.0, 5.0, 2.0, 1.0, 0.0]
            
            for beta in betas:
                if time.time() - start_time > time_limit - 10:
                    break
                res = minimize(lambda lp: objective(lp, beta), current_logP, method='L-BFGS-B', 
                              options={'maxiter': 500, 'ftol': 1e-11, 'gtol': 1e-10})
                
                new_score = -res.fun
                if new_score >= best_score:
                    best_score = new_score
                    current_logP = res.x.copy()
                    z = current_logP - np.max(current_logP)
                    P_flat = np.exp(z)
                    P_flat = P_flat / np.sum(P_flat)
                    best_P = P_flat.reshape(10, 10)
                else:
                    # If score dropped, keep the previous best but maybe try again with current
                    # Actually, if beta decreases, the objective becomes "harder" (less smooth).
                    # A drop in score when beta->0 is expected if the soft max was overestimating.
                    # We should compare against the true score (beta=0) at the end.
                    pass
        except Exception:
            pass

    # Final verification and exact calculation
    # Re-evaluate the best candidate with beta=0 to get the true score
    final_score = compute_ratio_from_logP(best_logP, 0.0)
    
    # If the final score is significantly lower, it might be due to numerical issues or bad convergence.
    # But we trust the optimization.
    
    # Convert final logP to P
    z = best_logP - np.max(best_logP)
    P_flat = np.exp(z)
    total = np.sum(P_flat)
    if total < 1e-15:
        best_P = np.ones((10, 10)) / 100.0
    else:
        best_P = (P_flat / total).reshape(10, 10)
    
    # Ensure validity constraints
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s < 1e-9:
        best_P = np.ones((10, 10)) / 100.0
    else:
        best_P = best_P / s
        
    return best_P
