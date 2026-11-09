"""Rewritten acquisition for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling-inspired acquisition: sample from the GP posterior
    at each candidate and score by negative sampled value (lower is better).
    This naturally balances exploration (high sigma -> wide sample range)
    and exploitation (low mu -> lower expected samples).
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    t0 = time.time()
    deadline = 325.0  # Leave 5s margin before 330s kill
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Thompson Sampling: draw one sample from N(mu, sigma^2) for each candidate
    # Use a fixed seed derived from f_best to get deterministic-ish behavior
    # that changes as f_best changes, giving natural diversity across iterations
    rng = np.random.default_rng(seed=int(abs(hash((f_best, len(mu)))) % (2**32)))
    samples = mu + sigma_safe * rng.standard_normal(size=mu.shape)
    
    # Score: negative sampled value (higher score = lower sampled value = better)
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    return scores
# EVOLVE-BLOCK-END
