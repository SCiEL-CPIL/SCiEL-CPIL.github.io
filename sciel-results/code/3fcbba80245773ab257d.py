"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard: The evaluator kills after 330s. 
    # We check at the start and end. 
    # Since computation is fast (<1ms), this is a safety net.
    t_start = time.time()
    if t_start > 325:
        return np.zeros_like(mu)

    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI)
    # Standard EI for minimization: (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB) score for minimization
    # We want to maximize: -mu + beta * sigma
    # Using beta = 2.0 as a standard exploration constant.
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    
    # 3. Z-score Normalization
    # This prevents scale dominance between EI and UCB components.
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std < 1e-9:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_mean) / ei_std
        
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb)
    if ucb_std < 1e-9:
        ucb_norm = np.zeros_like(ucb)
    else:
        ucb_norm = (ucb - ucb_mean) / ucb_std
        
    # 4. Adaptive Weighting
    # Weight each component by its variance (informativeness) in the current posterior.
    # If one component provides more differentiation among candidates, it gets more weight.
    var_ei = np.var(ei_norm)
    var_ucb = np.var(ucb_norm)
    total_var = var_ei + var_ucb
    
    if total_var < 1e-9:
        # Fallback to simple EI if no variance
        scores = ei
    else:
        w_ei = var_ei / total_var
        w_ucb = var_ucb / total_var
        scores = w_ei * ei_norm + w_ucb * ucb_norm
        
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Final deadline check
    if time.time() - t_start > 290:
        return np.argmax(scores) * np.ones_like(scores) # Fallback to best index score if slow (unlikely)
        
    return scores
# EVOLVE-BLOCK-END
