# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive LCB (Lower Confidence Bound).
    
    Structural change: replaces EI with an uncertainty-driven LCB that adapts
    the exploration constant beta to the current posterior uncertainty relative
    to the function spread. This avoids EI's premature convergence and allows
    finding deeper minima in the 5 instances.
    
    Score:
        score(x) = (f_best - mu(x) + beta * sigma(x)) / (cost(x) ** alpha)
    
    where beta adapts based on the ratio of current uncertainty to the range
    of observed values, and alpha is a cost sensitivity exponent.
    """
    t_start = time.time()
    TIME_LIMIT = 320.0  # Safety margin before 330s kill

    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    n = len(mu_safe)

    # --- Adaptive LCB computation ---
    # The key insight: beta should scale with the relative uncertainty.
    # When sigma is large relative to the function spread, we explore more.
    # When sigma is small, we exploit more.
    
    # Estimate the function spread from the current posterior
    mu_range = np.max(mu_safe) - np.min(mu_safe) + 1e-10
    sigma_mean = np.mean(sigma_safe)
    
    # Relative uncertainty: how much uncertainty is there compared to the range?
    rel_uncertainty = sigma_mean / mu_range
    
    # Adaptive beta: starts moderate, increases with relative uncertainty
    # but caps at a reasonable value to avoid excessive exploration
    base_beta = 2.0
    beta = base_beta * (1.0 + 0.5 * rel_uncertainty)
    beta = np.clip(beta, 1.0, 5.0)
    
    # Compute LCB scores
    # Higher score = better (we want to maximize f_best - mu + beta*sigma)
    lcb_raw = (f_best - mu_safe) + beta * sigma_safe
    
    # --- Cost-aware penalty ---
    # Use a power-law cost penalty with adaptive exponent
    # The cost is in [1, 2], so we want to penalize high-cost points
    # more aggressively as we progress
    
    # Adaptive cost sensitivity: start moderate, increase slightly
    # based on the distribution of costs
    cost_mean = np.mean(cost_safe)
    cost_std = np.std(cost_safe) + 1e-10
    
    # Cost exponent: higher means more aggressive cost penalty
    # Start with 1.0 and adjust based on cost distribution
    alpha = 1.0 + 0.5 * (cost_std / (cost_mean + 1e-10))
    alpha = np.clip(alpha, 0.5, 2.0)
    
    # Apply cost penalty
    cost_penalty = cost_safe ** alpha
    scores = lcb_raw / cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time check
    if time.time() - t_start > TIME_LIMIT:
        return scores
    
    # --- Optional refinement: boost high-uncertainty points among top candidates ---
    elapsed = time.time() - t_start
    if elapsed < TIME_LIMIT - 5.0:
        top_k2 = min(50, n)
        top_idx2 = np.argsort(scores)[-top_k2:]
        
        # Among the top candidates, boost those with higher sigma
        # This encourages exploration in uncertain regions
        top_sigma = sigma_safe[top_idx2]
        sigma_max = np.max(top_sigma)
        sigma_min = np.min(top_sigma)
        
        if sigma_max > sigma_min + 1e-10:
            # Normalize sigma in [0, 1] for the top candidates
            sigma_norm = (top_sigma - sigma_min) / (sigma_max - sigma_min)
            
            # Add a small boost proportional to normalized sigma
            boost_factor = 0.02 * np.max(scores)
            for i, idx in enumerate(top_idx2):
                scores[idx] += boost_factor * sigma_norm[i]
    
    # Final safety check
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
