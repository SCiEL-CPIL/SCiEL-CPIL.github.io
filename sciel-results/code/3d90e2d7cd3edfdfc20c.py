"""Thompson Sampling acquisition for cost-aware Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function.
    
    Samples a single objective value from the Gaussian posterior N(mu, sigma^2)
    for each candidate, then normalizes by the evaluation cost to balance
    exploration and exploitation.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean of the objective at each candidate point.
    sigma : array-like, shape (2000,)
        Posterior standard deviation of the objective at each candidate point.
    f_best : float
        Lowest observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Thompson Sampling scores; higher values are preferred for next evaluation.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative and add small epsilon for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Sample a single objective value from the Gaussian posterior for each candidate
    # Using a fixed seed per call is not appropriate; we use random sampling.
    # However, for determinism in testing, we could use a fixed random state,
    # but Thompson Sampling inherently uses randomness.
    # We'll use numpy's global random state.
    samples = mu + sigma * np.random.randn(mu.shape[0])
    
    # For minimization, lower sampled values are better.
    # We want higher scores to mean "evaluate next", so we negate the samples.
    # Then normalize by cost to account for cost-awareness.
    scores = -samples / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
