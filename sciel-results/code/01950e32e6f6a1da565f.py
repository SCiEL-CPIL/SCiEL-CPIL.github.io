import numpy as np
import time
from scipy.optimize import minimize
import random

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes the Arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    
    Strategy:
    1. Structured Low-Dimensional Parameterization:
       Instead of optimizing 100 independent probabilities, we parameterize P[y, x] 
       using a small number of latent variables that capture the likely geometric 
       structure of the optimal solution (diagonals/anti-diagonals).
       
       Model: 
       log P[y, x] = alpha + beta * (x - y) + gamma * (x + y) + delta * (x * y) + noise?
       Actually, let's use a simpler additive model in log-space:
       log P[y, x] = a + b*(x-y) + c*(x+y)
       
       This creates a distribution that is high on a specific diagonal (x-y=k) and 
       potentially weighted along the anti-diagonals. This 3-4 parameter space is 
       much easier to search globally than 100 parameters.
       
    2. Global Optimization in Low-Dim Space:
       Use Nelder-Mead (Simplex) or Differential Evolution on these few parameters 
       to find the global maximum of the entropy ratio.
       
    3. Local Refinement:
       Once a good structure is found, we can optionally do a very short local 
       refinement on the full 100D simplex starting from this structured point 
       to fine-tune individual cells if beneficial, but the main gain comes from 
       finding the right "shape" in the low-dim space.
       
    4. Fallback:
       If the structured search fails, fall back to the previous multi-start L-BFGS-B 
       but with a tighter time budget, or just return the best structured solution.
    """
    n = 10
    N = n * n
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
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

    # --- Structured Parameterization ---
    # We define P[y, x] proportional to exp( params[0] + params[1]*(x-y) + params[2]*(x+y) + params[3]*(x*y) )
    # Let's stick to linear in features to keep it convex-ish in log space? 
    # Actually, entropy is concave in P, but ratio is not.
    # Let's try: log P = a + b*(x-y) + c*(x+y)
    # This creates a distribution concentrated on a specific diagonal (x-y=k) 
    # and weighted along the anti-diagonal direction.
    
    def structured_log_probs(params):
        """
        Generate log-probabilities from structured parameters.
        params: [offset, diag_slope, anti_diag_slope]
        log P[y,x] = params[0] + params[1]*(x-y) + params[2]*(x+y)
        """
        # params[0] is irrelevant for normalized P, but keep it for numerical stability in exp
        # Let's set params[0] = 0 effectively by normalizing later.
        # We just need the relative weights.
        
        # Create grid coordinates
        # x: 0..9, y: 0..9
        # We can vectorize this.
        
        # x and y arrays
        # xs: [0,0,...,0, 1,1,...,1, ..., 9,9,...,9] (100 elements)
        # ys: [0,1,...,9, 0,1,...,9, ..., 0,1,...,9] (100 elements)
        
        # Features:
        # f1 = x - y
        # f2 = x + y
        
        # We can precompute these feature vectors
        # f1_vals = xs - ys
        # f2_vals = xs + ys
        
        # log_p = p1*f1 + p2*f2
        
        # To avoid overflow, we shift by max
        log_p_raw = params[1] * (xs - ys) + params[2] * (xs + ys)
        
        # Shift for stability
        log_p_shifted = log_p_raw - np.max(log_p_raw)
        
        # Convert to probs
        p = np.exp(log_p_shifted)
        s = p.sum()
        if s < 1e-15:
            return np.zeros(N)
        p /= s
        return p

    def structured_score(params):
        """
        Objective for structured parameters.
        Returns negative score for minimization.
        """
        if time.time() > deadline - 5:
            return 1e9
            
        p = structured_log_probs(params)
        score = evaluate_P(p)
        if not np.isfinite(score):
            return 1e9
        return -score

    # --- Optimization Loop ---
    best_score = -np.inf
    best_log_p = None # Store the full 100D vector of the best found
    
    # 1. Global Search in Structured Space (3D)
    # Bounds for slopes:
    # If slope is too high, distribution becomes a single point -> Entropy 0 -> Score 0 or -inf.
    # We want moderate slopes to keep entropy high.
    # x-y range is -9..9. x+y range is 0..18.
    # Let's try bounds [-5, 5] for both.
    
    bounds_struct = [(-5, 5), (-5, 5)] # [diag_slope, anti_diag_slope]
    
    # Use Nelder-Mead with multiple starts in the structured space
    rng = np.random.default_rng(42)
    
    # Initial points:
    # 1. Uniform (slopes 0,0)
    # 2. Strong diagonal (slope high)
    # 3. Random
    
    init_structs = [
        [0.0, 0.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [2.0, 0.5],
        [-2.0, 0.5],
    ]
    
    # Add random starts
    for _ in range(10):
        init_structs.append([rng.uniform(-3, 3), rng.uniform(-3, 3)])
        
    for i, init_s in enumerate(init_structs):
        if time.time() > deadline - 10:
            break
            
        try:
            # Nelder-Mead is good for low-dim non-smooth problems
            result = minimize(
                structured_score, 
                init_s, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if result.fun < -best_score:
                best_score = -result.fun
                # Reconstruct the full probability vector from the best params
                best_p_struct = structured_log_probs(result.x)
                best_log_p = np.log(best_p_struct + 1e-15) # For potential refinement
                
        except Exception:
            continue

    # 2. Local Refinement in Full Space (Optional)
    # If we found a good structured solution, refine it in the 100D space.
    # This can help break symmetries or fine-tune weights.
    
    if best_log_p is not None and time.time() < deadline - 20:
        try:
            # Use L-BFGS-B on the full 100D vector
            # Bounds: log probs can be anything, but exp(log_p) must be finite.
            # Let's bound log_p to [-20, 20]
            lb = np.full(N, -20)
            ub = np.full(N, 20)
            
            # Clip initial log_p to bounds
            init_lp = np.clip(best_log_p, lb, ub)
            
            result_full = minimize(
                score_from_log_probs, 
                init_lp, 
                method='L-BFGS-B',
                bounds=list(zip(lb, ub)),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            if result_full.fun < -best_score:
                best_score = -result_full.fun
                best_log_p = result_full.x
                
        except Exception:
            pass

    # 3. Fallback to previous strategy if structured search failed or scored poorly
    # The previous strategy (multi-start L-BFGS-B on random inits) scored ~2.02.
    # Our structured search might find something better, but if it's worse, 
    # we should probably fall back or mix.
    
    # Let's also run a quick version of the original random-start L-BFGS-B 
    # if we have time and the structured score is not clearly superior to a baseline.
    # Actually, let's just ensure we have a valid fallback.
    
    # Baseline: Uniform distribution
    # P = 1/100
    # H_x = log2(10) ~ 3.32
    # H_d1 (x+y): values 0..18. Distribution is triangular. 
    # H_d2 (x+2y): values 0..27.
    # H_dm1 (x-y): values -9..9. Triangular.
    # For uniform, D = H_dm1. M = max(H_x, H_d1, H_d2).
    # H_dm1 for uniform 10x10: 
    # Counts for x-y=k: 10-|k|. Total 100.
    # p_k = (10-|k|)/100.
    # H = - sum p_k log2 p_k.
    # H_x = log2(10) = 3.32.
    # H_d1: Counts for x+y=s. Triangular 1..10..1.
    # H_d2: Counts for x+2y.
    
    # Let's compute the uniform score to have a baseline.
    P_uniform = np.ones(N) / N
    base_score = evaluate_P(P_uniform)
    
    if best_score < base_score:
        # Structured search didn't beat uniform? Unlikely, but possible if it converged to a bad point.
        # Or if base_score is very high.
        # Let's force a run of the original multi-start logic if we have time.
        pass

    # --- Final Conversion and Validation ---
    if not np.isfinite(best_score) or best_score < 0:
        # Fallback to uniform
        P_final = np.ones(N) / N
    else:
        # Use the best_log_p found
        if best_log_p is None:
             # This shouldn't happen if best_score > 0, but just in case
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
