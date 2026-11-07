"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    Uses Upper Confidence Bound (UCB) for minimization.
    Score = -mu + beta * sigma. Higher score is preferred.
    """
    # Beta tuned based on knowledge that lower beta (1.0 vs 2.5) improved performance.
    beta = 1.0
    
    # Calculate UCB scores for minimization
    # We maximize this score to select the next point.
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite to avoid invalid results
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
