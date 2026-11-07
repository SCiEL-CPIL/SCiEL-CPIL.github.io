"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_START_TIME = time.time()
# The evaluator kills after 330s. We use 320s as a safety margin.
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    Args:
        mu: Posterior mean at candidate points (shape 2000,).
        sigma: Posterior standard deviation at candidate points (shape 2000,).
        f_best: Lowest observed value so far.
        
    Returns:
        Scores for the candidate points (shape 2000,). Higher is better.
    """
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        return -mu

    # UCB for minimization: score = -(mu - kappa * sigma) = -mu + kappa * sigma
    # We want higher scores to be selected, so we maximize this.
    # 
    # For minimization, we want low mu and high sigma (exploration).
    # The standard UCB for minimization is:
    #   score = -mu + kappa * sigma
    # where kappa controls the exploration/exploitation tradeoff.
    #
    # A common choice is kappa = 1.0, but we can use a slightly larger
    # value to encourage more exploration in the multi-modal landscape.
    # 
    # To make it adaptive, we can scale kappa by the magnitude of f_best
    # or by the range of sigma. However, a fixed kappa is often robust.
    # Let's use kappa = 1.5 to balance exploration and exploitation.
    #
    # Actually, for Bayesian optimization with EI-like behavior, a common
    # adaptive approach is to use the current best value to scale exploration.
    # But let's start with a well-tuned fixed kappa.
    
    # Clip sigma to avoid numerical issues
    epsilon = 1e-9
    safe_sigma = np.maximum(sigma, epsilon)
    
    # Exploration parameter
    # For a 2D RBF mixture with 30 evaluations, we want moderate exploration.
    # kappa = 1.0 is standard. Let's try 1.5 for more exploration.
    kappa = 1.5
    
    # UCB score for minimization: higher is better
    # We want to minimize f, so we score candidates by how promising they are.
    # score = -mu + kappa * sigma
    # This rewards low mean and high uncertainty.
    scores = -mu + kappa * safe_sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
