"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) / Lower Confidence Bound (LCB) for minimization.
    # We want to maximize the score to select the next point.
    # For minimization, we prefer points with low predicted mean (mu) and high uncertainty (sigma).
    # The LCB is mu - beta * sigma. Minimizing LCB is equivalent to maximizing -LCB.
    # Score = -mu + beta * sigma
    
    beta = 2.0
    
    # Ensure inputs are clean
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    sigma_safe = np.where(np.isfinite(sigma) & (sigma >= 0), sigma, 0.0)
    
    # Calculate score
    scores = -mu_safe + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
