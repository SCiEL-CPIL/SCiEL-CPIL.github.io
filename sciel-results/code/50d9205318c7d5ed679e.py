import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes the Arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    Strategy:
    1. Generate diverse structured initializations (diagonals, corners, uniforms).
    2. Perform local refinement using L-BFGS-B on log-probabilities from each start.
    3. Return the best valid result found within the time budget.
    This avoids the overhead and potential stagnation of global stochastic search
    by leveraging the smoothness of the entropy landscape in log-space.
    """
    n = 10
    N = n * n
    start_time = time.time()
    # Keep a safety margin for final validation and return
    deadline = start_time + 600
    
    # Pre-compute indices for entropy calculation
    ys = np.repeat(np.arange(n), n)
    xs = np.tile(np.arange(n), n)
    
    # Indices for binning
    idx_dm1 = (xs - ys + n - 1).astype(np.int32)       # x - y + 9 -> 0..18
    idx_x   = xs.astype(np.int32)                      # x -> 0..9
    idx_d1  = (xs + ys).astype(np.int32)               # x + y -> 0..18
    idx_d2  = (xs + 2 * ys).astype(np.int32)           # x + 2y -> 0..27
    
    bin_sizes = {
        'dm1': 2 * n - 1,
        'x': n,
        'd1': 2 * n - 1,
        'd2': 3 * n - 1
    }

    def entropy_from_counts(counts):
        """Calculate Shannon entropy (bits) from a count/probability vector."""
        mask = counts > 1e-15
        if not np.any(mask):
            return 0.0
        p = counts[mask]
        p_sum = p.sum()
        if p_sum < 1e-15:
            return 0.0
        p = p / p_sum
        return float(-np.sum(p * np.log2(p)))

    def evaluate_P(P_flat):
        """
        Evaluate the score for a flattened probability vector P_flat.
        """
        if not np.all(np.isfinite(P_flat)):
            return -np.inf
        
        # Clip and normalize
        P_norm = np.clip(P_flat, 0, None)
        s = P_norm.sum()
        if s < 1e-9:
            return -np.inf
        P_norm /= s

        c_dm1 = np.bincount(idx_dm1, weights=P_norm, minlength=bin_sizes['dm1'])
        c_x   = np.bincount(idx_x,   weights=P_norm, minlength=bin_sizes['x'])
        c_d1  = np.bincount(idx_d1,  weights=P_norm, minlength=bin_sizes['d1'])
        c_d2  = np.bincount(idx_d2,  weights=P_norm, minlength=bin_sizes['d2'])

        H_dm1 = entropy_from_counts(c_dm1)
        H_x   = entropy_from_counts(c_x)
        H_d1  = entropy_from_counts(c_d1)
        H_d2  = entropy_from_counts(c_d2)

        D = H_dm1
        M = max(H_x, H_d1, H_d2)
        
        if M < 1e-9:
            return -np.inf
            
        return D / M

    def score_from_log_probs(log_p):
        """
        Objective function for minimization.
        Input: log_p (unnormalized log-probabilities)
        Output: Negative score
        """
        log_p_shifted = log_p - np.max(log_p)
        p = np.exp(log_p_shifted)
        s = p.sum()
        if s < 1e-15:
            return 1e9 
        p = p / s
        
        score = evaluate_P(p)
        if not np.isfinite(score):
            return 1e9
        return -score

    # --- Generate Initializations ---
    # We want a mix of structured and random starts to cover different basins of attraction.
    inits = []
    
    # 1. Uniform
    inits.append(np.zeros(N))
    
    # 2. Diagonal concentration (maximizes H_{-1} potentially)
    # P[y, x] high when x == y
    diag = np.zeros(N)
    for i in range(n):
        diag[i*n + i] = 1.0
    inits.append(np.log(diag + 1e-10))
    
    # 3. Anti-diagonal
    anti = np.zeros(N)
    for i in range(n):
        anti[i*n + (n-1-i)] = 1.0
    inits.append(np.log(anti + 1e-10))
    
    # 4. Row concentration (x constant)
    row = np.zeros(N)
    for j in range(n):
        row[j*n + j] = 1.0 # This is same as diag. Let's do row 0
    row = np.zeros(N)
    row[0:n] = 1.0
    inits.append(np.log(row + 1e-10))
    
    # 5. Column concentration (y constant)
    col = np.zeros(N)
    col[0::n] = 1.0
    inits.append(np.log(col + 1e-10))
    
    # 6. Corner (0,0)
    corner = np.zeros(N)
    corner[0] = 1.0
    inits.append(np.log(corner + 1e-10))
    
    # 7. Random starts
    rng = np.random.default_rng(42)
    for _ in range(20):
        # Random log-probabilities in a reasonable range
        lp = rng.uniform(-5, 5, size=N)
        inits.append(lp)
        
    # 8. Structured random: high on diagonals x + ky = c
    for k in range(1, 4):
        for c in range(-n, n):
            lp = np.zeros(N)
            for y in range(n):
                x = c - k*y
                if 0 <= x < n:
                    lp[y*n + x] = 1.0
            if np.any(lp > 0):
                 inits.append(np.log(lp + 1e-10))

    # --- Optimization Loop ---
    best_score = -np.inf
    best_log_p = np.zeros(N)
    
    # Bounds for log-probs to prevent overflow/underflow issues in exp
    lb = np.full(N, -20)
    ub = np.full(N, 20)
    
    # L-BFGS-B options
    # maxiter: limit per run to ensure we can try multiple starts
    # ftol/gtol: high precision
    options = {'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10}
    
    for i, init_lp in enumerate(inits):
        # Check time
        if time.time() > deadline - 5:
            break
            
        try:
            # Clip init to bounds
            init_lp_clipped = np.clip(init_lp, lb, ub)
            
            result = minimize(
                score_from_log_probs, 
                init_lp_clipped, 
                method='L-BFGS-B',
                bounds=list(zip(lb, ub)),
                options=options
            )
            
            if result.fun < -best_score:
                best_score = -result.fun
                best_log_p = result.x
                
        except Exception:
            continue
            
    # --- Final Conversion and Validation ---
    if not np.isfinite(best_score) or best_score < 0:
        # Fallback to uniform if optimization failed completely
        P_final = np.ones(N) / N
    else:
        log_p_shifted = best_log_p - np.max(best_log_p)
        P_final = np.exp(log_p_shifted)
        P_final = np.clip(P_final, 0, None)
        s = P_final.sum()
        if s > 1e-9:
            P_final /= s
        else:
            P_final = np.ones(N) / N
            
    P_final = P_final.reshape((n, n))
    
    # Final safety check
    if not np.all(np.isfinite(P_final)) or P_final.shape != (n, n):
        return np.ones((n, n)) / N
        
    return P_final

if __name__ == "__main__":
    pass
