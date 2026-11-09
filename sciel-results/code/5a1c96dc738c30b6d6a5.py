import numpy as np
import time
import scipy.special as sp

_start_time = time.time()
TIME_LIMIT = 300.0

_acq_call_count = 0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Draws a single sample from the posterior predictive distribution at all
    2000 candidates and returns these samples as scores. This transforms the
    search into a randomized stochastic optimization process that naturally
    balances exploration and exploitation without manual tuning.
    
    Parameters:
    mu (np.array): Mean of the GP posterior, shape (2000,)
    sigma (np.array): Standard deviation of the GP posterior, shape (2000,)
    f_best (float): Best observed function value so far.
    
    Returns:
    np.array: Scores for the 2000 candidates, shape (2000,). Higher is better.
    """
    global _acq_call_count
    _acq_call_count += 1
    
    # Deadline guard
    if time.time() - _start_time > TIME_LIMIT:
        # Fallback to exploitation: return negative mean (lower mean = higher score)
        return -np.asarray(mu, dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Thompson Sampling: draw a single sample from the posterior predictive
    # At each candidate x_i, the posterior is N(mu_i, sigma_i^2)
    # We want to minimize, so we score by the negative of the sample
    # sample_i = mu_i + sigma_i * z_i where z_i ~ N(0,1)
    # Higher score means we should evaluate next, so we return -sample_i
    # This means we pick the candidate with the lowest predicted value
    
    # Generate standard normal samples
    z = np.random.standard_normal(len(mu))
    
    # Compute posterior samples
    samples = mu + sigma * z
    
    # Return negative samples so that higher scores correspond to lower predicted values
    score = -samples
    
    # Ensure finite scores
    score[~np.isfinite(score)] = 0.0
    
    return score
