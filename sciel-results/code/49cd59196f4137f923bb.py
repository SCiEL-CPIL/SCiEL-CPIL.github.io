"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    to balance exploitation and exploration.
    
    mu: mean of GP posterior (2000,)
    sigma: std of GP posterior (2000,)
    f_best: best observed value so far (scalar)
    
    Returns: scores (2000,), higher means evaluate next
    """
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Time guard: if we're running long, use a fast deterministic method
    # We track time across calls using a module-level variable
    global _call_start_time, _elapsed
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
        acquisition._call_count = 0
    acquisition._call_count += 1
    elapsed = time.time() - acquisition._start_time
    
    # If we've been running for more than 300s, use the fastest possible method
    if elapsed > 280:
        # Fast fallback: simple LCB with moderate beta
        beta = 2.5
        scores = beta * sigma - mu
        return scores
    
    # Compute EI (Expected Improvement for minimization)
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal
    from scipy.stats import norm
    
    # Avoid division by zero
    safe_sigma = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / safe_sigma
    
    # EI formula
    ei = (f_best - mu) * norm.cdf(z) + safe_sigma * norm.pdf(z)
    
    # UCB score for exploration (higher is better for exploration)
    # We want to explore regions with high uncertainty
    # Use a dynamic beta that decreases over time to shift from exploration to exploitation
    # Early on (few evaluations), explore more; later, exploit more
    # We estimate evaluation count from call count (each call = 1 evaluation step)
    n_evals = acquisition._call_count
    
    # Dynamic beta: start high, decrease over time
    # For 30 total evaluations, we want beta to go from ~4.0 to ~1.5
    if n_evals < 10:
        beta_ucb = 4.0
    elif n_evals < 20:
        beta_ucb = 3.0
    else:
        beta_ucb = 2.0
    
    ucb_score = beta_ucb * sigma
    
    # Combine EI and UCB
    # Normalize both to [0, 1] range for combination
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-10
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min + 1e-10
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # Weighted combination: more weight on EI as we progress
    if n_evals < 10:
        w_ei = 0.3
        w_ucb = 0.7
    elif n_evals < 20:
        w_ei = 0.5
        w_ucb = 0.5
    else:
        w_ei = 0.7
        w_ucb = 0.3
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
