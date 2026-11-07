"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses Thompson Sampling:
    Instead of computing a deterministic score like EI or UCB, we sample a
    function value from the posterior distribution at each candidate point.
    Specifically, we draw a single sample from a Normal distribution with
    mean mu and standard deviation sigma.
    
    Since the goal is to minimize the function, the candidate with the lowest
    sampled value is selected. To fit the interface which requires "higher is
    better" scores, we return the negative of the sampled values.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, keep 20s margin

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Thompson Sampling: Draw a sample from N(mu, sigma)
    # Using a single draw is efficient and provides the necessary stochasticity
    # for exploration.
    # np.random.normal is fast for arrays of this size.
    samples = np.random.normal(loc=mu, scale=sigma)
    
    # Check time, though this operation is very fast
    if time.time() > deadline:
        # Fallback to pure exploitation if something went wrong or took too long
        scores = -mu
    else:
        # We want to minimize the function value.
        # The evaluator picks the candidate with the HIGHEST score.
        # So we negate the sample: high score corresponds to low sampled value.
        scores = -samples
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=-1e30, posinf=1e30, neginf=-1e30)
    
    # Clip to ensure finite bounds as per contract
    max_float = 1e30
    scores = np.clip(scores, -max_float, max_float)
    
    return scores
# EVOLVE-BLOCK-END
