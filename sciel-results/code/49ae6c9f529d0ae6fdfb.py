import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes the Arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    Strategy: 
    1. Differential Evolution on log-probabilities (robust global search).
    2. L-BFGS-B polish on the best result from DE (local refinement).
    """
    n = 10
    N = n * n
    start_time = time.time()
    # Leave 20s buffer for overhead, return, and potential final checks
    deadline = start_time + 610
    
    # Pre-compute indices for entropy calculation
    # P[y, x] -> flat index y*n + x
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
        Assumes P_flat is non-negative and sums to ~1.
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
        Objective function for optimization.
        Input: log_p (unnormalized log-probabilities)
        Output: Negative score (for minimization)
        """
        # Softmax to get probabilities
        # Subtract max for numerical stability
        log_p_shifted = log_p - np.max(log_p)
        p = np.exp(log_p_shifted)
        s = p.sum()
        if s < 1e-15:
            return 1e9 # Large penalty
        p = p / s
        
        score = evaluate_P(p)
        if not np.isfinite(score):
            return 1e9
        return -score

    # --- Phase 1: Differential Evolution ---
    # Bounds for log-probabilities. 
    # We can use a wide range, e.g., [-10, 10]. 
    # DE works on the raw parameter space, so we don't need strict simplex constraints 
    # because softmax handles the normalization.
    
    # Initial bounds
    bounds = [(-10, 10) for _ in range(N)]
    
    # Use a relatively small population to allow many generations within time limit
    # DE is stochastic, so we fix seed for reproducibility if needed, but default is fine
    # maxiter is capped by time anyway via the callback or just running a fixed amount
    # Let's run DE for a fixed number of iterations or until time is up.
    
    # We'll run DE for 100 iterations. If it's too slow, we'll catch the time out.
    # DE is generally fast for 100 dims.
    
    # To ensure we don't exceed time, we'll wrap the DE call or use a callback.
    # Simpler: Run DE with a reasonable maxiter and check time after.
    # If DE is too slow, we fallback to the best found.
    
    try:
        # seed=42 for reproducibility in testing, but random is fine
        # tol=1e-5, maxiter=100
        # popsize=15 (default)
        result_de = differential_evolution(
            score_from_log_probs, 
            bounds, 
            maxiter=50, 
            popsize=20,
            seed=42,
            tol=1e-4,
            polish=False # We will polish manually
        )
    except Exception as e:
        # If DE fails or times out unexpectedly, we need a fallback
        # For now, assume it finishes within 540s. 50 iters * 20 pop * 100 dims is small.
        pass

    best_log_p = result_de.x
    best_score_de = -result_de.fun
    
    # Convert best log_p to P
    log_p_shifted = best_log_p - np.max(best_log_p)
    P_best = np.exp(log_p_shifted)
    P_best /= P_best.sum()
    
    # --- Phase 2: Local Refinement with L-BFGS-B ---
    # Use the result from DE as starting point for L-BFGS-B
    # L-BFGS-B is good for smooth parts of the landscape.
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < deadline - 30: # Leave 30s for L-BFGS and final checks
        
        # L-BFGS-B on log-probabilities
        # We need to handle the fact that L-BFGS-B might step outside "reasonable" bounds,
        # but softmax is defined everywhere, so no bounds are strictly needed, 
        # though it helps to prevent extreme values.
        
        # Let's use bounds to keep log_p in a reasonable range, e.g., [-20, 20]
        lb = np.full(N, -20)
        ub = np.full(N, 20)
        
        try:
            result_lbfgs = minimize(
                score_from_log_probs, 
                best_log_p, 
                method='L-BFGS-B',
                bounds=list(zip(lb, ub)),
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result_lbfgs.success or result_lbfgs.fun < result_de.fun:
                best_log_p = result_lbfgs.x
                best_score = -result_lbfgs.fun
            else:
                best_score = best_score_de
                
        except Exception:
            best_score = best_score_de

    else:
        best_score = best_score_de

    # --- Final Conversion and Validation ---
    log_p_shifted = best_log_p - np.max(best_log_p)
    P_final = np.exp(log_p_shifted)
    P_final = np.clip(P_final, 0, None)
    s = P_final.sum()
    if s > 1e-9:
        P_final /= s
    else:
        # Fallback to uniform if something went wrong
        P_final = np.ones(N) / N
        
    P_final = P_final.reshape((n, n))
    
    # Final safety check
    if not np.all(np.isfinite(P_final)) or P_final.shape != (n, n):
        return np.ones((n, n)) / N
        
    return P_final

if __name__ == "__main__":
    pass
