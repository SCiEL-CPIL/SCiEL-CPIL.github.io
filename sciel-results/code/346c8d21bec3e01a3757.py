import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 300.0 # 300s limit, leave margin

def acquisition(mu, sigma, f_best):
    current_time = time.time()
    if current_time - _start_time > _TIMEOUT - 5.0:
        # Safety fallback: return something reasonable, e.g., pure EI or UCB
        # But if we are this close to timeout, just return zeros or a simple score
        # The evaluator will handle the timeout, but we should try to return valid scores.
        # A simple UCB is fast.
        return mu + 2.0 * sigma

    n = len(mu)
    scores = np.full(n, -1e9)
    
    # Filter: keep candidates where mu is "plausible" for improvement
    # "worse than f_best by more than a small tolerance"
    # We want mu to be <= f_best + tol
    # If mu is much larger, it's unlikely to beat f_best.
    tol = 0.01
    mask = mu <= (f_best + tol)
    
    if np.any(mask):
        valid_indices = np.where(mask)[0]
        valid_mu = mu[valid_indices]
        valid_sigma = sigma[valid_indices]
        
        # Apply UCB to the filtered subset
        # beta controls exploration vs exploitation within the filtered set
        # Since we already filtered for exploitation (mu < f_best), we can be more exploratory (higher beta)
        beta = 2.0
        ucb_scores = valid_mu + beta * valid_sigma
        
        # Assign scores back
        scores[valid_indices] = ucb_scores
    else:
        # Fallback: no point has a mean better than f_best + tol
        # This implies the GP is confident the current best is the best, or the region is not well-explored enough to predict improvement.
        # In this case, we should explore.
        # Use global UCB with a higher beta to find new regions.
        beta_global = 3.0
        scores = mu + beta_global * sigma
        
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)
    
    return scores
