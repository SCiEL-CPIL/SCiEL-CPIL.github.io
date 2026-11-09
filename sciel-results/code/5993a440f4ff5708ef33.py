"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guard
_START_TIME = time.time()


def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores using a Thompson Sampling approach.
    Samples function values from the posterior for each candidate and
    returns these samples as scores. This naturally balances exploration
    and exploitation without requiring fixed parameters.
    
    For minimization, we want to sample from the posterior and select
    the candidate with the lowest sampled value. Since the evaluator
    picks the candidate with the HIGHEST score, we negate the samples.
    """
    # Deadline guard: 330s limit. Return immediately if close to limit.
    # 10s safety margin.
    if time.time() - _START_TIME > 320:
        return np.zeros_like(mu)
    
    # Clip sigma to avoid division by zero and ensure positive values
    sigma = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: draw one sample from the posterior for each candidate
    # sample = mu + sigma * z where z ~ N(0, 1)
    rng = np.random.default_rng()
    z = rng.standard_normal(size=mu.shape)
    
    # Sampled function values from the posterior
    samples = mu + sigma * z
    
    # For minimization, we want the lowest sampled value to be selected.
    # Since the evaluator picks the highest score, we negate the samples.
    # This way, the candidate with the lowest predicted value gets the highest score.
    score = -samples
    
    # Ensure scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
