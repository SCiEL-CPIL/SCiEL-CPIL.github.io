"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function with adaptive exploitation/exploration balance.
    
    EI is deterministic and well-suited for the structured search here. We blend EI with a
    small UCB term to ensure exploration of uncertain regions, especially early on.
    """
    # Ensure inputs are finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Standardize for EI computation
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal
    
    delta = f_best - mu
    z = delta / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Compute Phi(z) and phi(z)
    from scipy.stats import norm
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement
    ei = delta * Phi_z + sigma * phi_z
    
    # Handle edge case: if mu >= f_best, EI should be 0 (no improvement possible)
    ei = np.where(delta > 0, ei, 0.0)
    
    # Add a small UCB component for exploration
    # beta adapts: larger when f_best is large (early), smaller when f_best is small (late)
    f_best_clamped = max(f_best, 0.0)
    
    # beta schedule: 
    # f_best >= 1.0: beta = 2.0 (significant exploration)
    # f_best = 0.0: beta = 0.5 (mostly exploitation)
    if f_best_clamped >= 1.0:
        beta = 2.0
    else:
        beta = 0.5 + 1.5 * f_best_clamped
    
    # UCB term: beta * sigma
    ucb_term = beta * sigma
    
    # Blend: score = EI + UCB
    # EI is already in objective units, UCB is also in objective units
    scores = ei + ucb_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
