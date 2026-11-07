"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We want to minimize f, so we look for low mu.
    # We add a bonus for high sigma (uncertainty) to encourage exploration.
    # Score = mu - beta * sigma. Higher score is better.
    # beta controls the exploration/exploitation trade-off.
    # A value around 2.0-3.0 is often effective for small budgets.
    beta = 2.5
    scores = mu - beta * sigma
    
    # Ensure finite scores as required by the contract
    # Replace any non-finite values with a very low score (bad)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
