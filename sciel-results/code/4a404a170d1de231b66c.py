import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Precompute shift indices for correlation
    # We need shifts k such that the correlation is valid and k != 0
    # In 'full' mode, index N-1 is shift 0.
    # We want max of c[k] for k != N-1.
    
    def get_raw_overlap(h):
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # Exclude shift 0 (index N-1)
        c[N-1] = -np.inf
        return np.max(c)

    def get_smooth_overlap(h, beta):
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        c[N-1] = -np.inf
        # Log-sum-exp trick for stability
        # max_c = np.max(c)
        # return max_c + np.log(np.sum(np.exp(c - max_c))) / beta
        # But we want the value itself scaled by beta in the exponent for the optimizer
        # Objective term: (1/beta) * log(sum(exp(beta * c_k)))
        # To avoid overflow, we can do:
        # M = np.max(c)
        # val = M + np.log(np.sum(np.exp(beta * (c - M)))) / beta
        # This is equivalent.
        max_c = np.max(c)
        if max_c == -np.inf:
            return 0.0
        # c - max_c is <= 0, so exp is stable
        sum_exp = np.sum(np.exp(beta * (c - max_c)))
        return max_c + np.log(sum_exp) / beta

    def objective(h, beta, lambda_pen):
        # h is flat array
        smooth_val = get_smooth_overlap(h, beta)
        sum_h = np.sum(h)
        penalty = lambda_pen * (sum_h - N/2)**2
        return smooth_val + penalty

    def valid_h(h):
        return np.all(h >= -1e-9) and np.all(h <= 1 + 1e-9) and abs(np.sum(h) - N/2) < 1e-6

    # Initial guess
    # Start with a random valid h
    def random_initial():
        h = np.random.rand(N)
        h *= (N/2) / np.sum(h)
        # Clip and fix
        for _ in range(10):
            mask = h > 1
            if not np.any(mask): break
            excess = np.sum(h[mask] - 1)
            h[mask] = 1
            non_mask = ~mask
            if np.sum(non_mask) == 0: break
            h[non_mask] -= excess / np.sum(non_mask)
            
        mask_low = h < 0
        if np.any(mask_low):
            deficit = np.sum(-h[mask_low])
            h[mask_low] = 0
            non_mask = ~mask_low
            h[non_mask] += deficit / np.sum(non_mask)
            
        diff = N/2 - np.sum(h)
        h += diff / N
        h = np.clip(h, 0, 1)
        diff = N/2 - np.sum(h)
        h += diff / N
        return h

    best_h = random_initial()
    best_score = get_raw_overlap(best_h)
    
    bounds = [(0, 1)] * N
    
    # Homotopy parameters
    betas = [1.0, 5.0, 20.0, 100.0, 500.0]
    lambda_pen = 100.0 # Large penalty to enforce sum constraint
    
    current_h = best_h.copy()
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
            
        # Optimize
        try:
            res = minimize(
                objective, 
                current_h, 
                args=(beta, lambda_pen),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            new_h = res.x
            # Ensure constraints are met (L-BFGS-B respects bounds, but sum might drift slightly if penalty isn't infinite)
            # We can project back to sum N/2 if needed, but penalty should handle it.
            # Let's check sum
            sum_new = np.sum(new_h)
            if abs(sum_new - N/2) > 1e-4:
                # Project
                diff = N/2 - sum_new
                new_h += diff / N
                new_h = np.clip(new_h, 0, 1)
                # Re-project sum if clip broke it
                # Simple iterative fix
                for _ in range(5):
                    diff = N/2 - np.sum(new_h)
                    if abs(diff) < 1e-8: break
                    new_h += diff / N
                    new_h = np.clip(new_h, 0, 1)
            
            new_score = get_raw_overlap(new_h)
            
            if new_score < best_score:
                best_score = new_score
                best_h = new_h.copy()
            
            current_h = new_h.copy()
            
        except Exception as e:
            pass
            
    # Final local polish with small random steps to fix sum/bounds if necessary
    # and to try to improve the raw score further since smooth approx might be slightly off
    # Use a few steps of SA on the result? No, keep it simple.
    # Just ensure the returned h is valid.
    
    # Final projection
    diff = N/2 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    # Fix sum again
    for _ in range(10):
        diff = N/2 - np.sum(best_h)
        if abs(diff) < 1e-12: break
        best_h += diff / N
        best_h = np.clip(best_h, 0, 1)
        
    return best_h
# EVOLVE-BLOCK-END
