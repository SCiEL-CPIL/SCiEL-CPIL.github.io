"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Use Upper Confidence Bound (UCB) for minimization.
    # We maximize the score: Score = beta * sigma - mu
    # This balances exploitation (low mu) and exploration (high sigma).
    
    # Exploration constant
    beta = 2.0
    
    # Ensure sigma is non-negative and avoid division issues (though not dividing here)
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate UCB scores
    scores = beta * sigma_safe - mu
    
    # Ensure output is finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
