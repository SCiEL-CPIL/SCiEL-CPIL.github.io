import numpy as np
import time
import scipy.special as sp

_start_time = time.time()
TIME_LIMIT = 300.0

_acq_call_count = 0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
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
        # Fallback to exploitation
        return -np.asarray(mu, dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Thompson Sampling: draw a sample from the posterior predictive distribution
    # For minimization, we want to find low values, so we sample from N(mu, sigma^2)
    # and return negative samples (higher score = lower predicted value)
    noise = np.random.standard_normal(len(mu))
    sample = mu - sigma * noise
    
    # Ensure finite scores
    sample[~np.isfinite(sample)] = 0.0
    
    return sample
