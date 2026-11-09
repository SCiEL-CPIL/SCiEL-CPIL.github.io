"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of cosine8 in 8D.
    Uses UCB (Upper Confidence Bound) with a multiplicative exponential cost gate.
    This shifts from the non-linear EI to a linear uncertainty exploration term,
    providing a different structural exploration strategy.
    """
    t_start = time.time()
    
    # Numerical safety
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate UCB for minimization
    # We want to maximize: (f_best - mu) + kappa * sigma
    # This represents the potential improvement plus an exploration bonus.
    # kappa controls the trade-off between exploitation (f_best - mu) and exploration (sigma).
    kappa = 2.0
    
    ucb_term = (f_best - mu) + kappa * sigma_safe
    
    # Ensure the base score is non-negative to avoid selecting points with negative expected value
    # unless the exploration term is very strong? 
    # Actually, if f_best - mu is very negative (mu >> f_best), and kappa*sigma is not enough, 
    # the term is negative. We should probably clamp to 0 to force focus on promising or very uncertain areas.
    base_score = np.maximum(ucb_term, 0.0)
    
    # Multiplicative cost gate
    # Keep k = 1.5 as it was effective in the parent
    k = 1.5
    gate = np.exp(-k * cost)
    
    score = base_score * gate
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard
    t_elapsed = time.time() - t_start
    if t_elapsed > 5.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
