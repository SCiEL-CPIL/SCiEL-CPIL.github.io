"""Rewritten program using Thompson Sampling for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition: draw a sample from the posterior at each
    candidate and return negative sampled values so that higher scores
    correspond to lower (better) function values.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Posterior mean at each candidate point.
    sigma : np.ndarray, shape (2000,)
        Posterior standard deviation at each candidate point.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Finite scores; higher means evaluate next.
    """
    # Create a deterministic random seed from the input data so that
    # the same inputs always produce the same output, but different
    # calls (with different mu/sigma) get different samples.
    # We hash a small portion of the inputs to get an integer seed.
    
    # Use a simple hash of the first few values to create a seed
    seed_material = (int(np.sum(mu[:10]) * 1e6) ^ 
                    int(np.sum(sigma[:10]) * 1e6) ^
                    int(abs(f_best) * 1e6))
    seed = seed_material % (2**32)
    
    rng = np.random.default_rng(seed)
    
    # Draw a single sample from N(mu, sigma^2) at each candidate
    samples = mu + sigma * rng.standard_normal(mu.shape[0])
    
    # Return negative samples so that lower function values (better)
    # get higher scores
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
