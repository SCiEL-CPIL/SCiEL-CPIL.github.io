import numpy as np
import time
import scipy.special as sp

_start_time = time.time()
TIME_LIMIT = 325.0

_acq_call_count = 0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with a time-adaptive
    exploration bonus. This approach has been shown to perform well for the
    Beale function, balancing exploitation of known good regions with
    exploration of uncertain regions.
    
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
    
    # Calculate progress (0 to 1)
    step = (_acq_call_count - 1) % 25
    progress = step / 25.0
    
    # Compute Expected Improvement (EI)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF using scipy.special
    cdf_z = 0.5 * sp.erfc(-z / np.sqrt(2.0))
    pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    # Standard EI formula
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Time-adaptive exploration weight.
    # We want to explore aggressively at the start (progress ~ 0)
    # and exploit at the end (progress ~ 1).
    # Using a power law decay for a smooth transition.
    # Start with higher exploration to escape local optima in the early stages.
    lambda_explore = 1.5 * np.power(1.0 - progress, 1.5)
    
    # Hybrid score: EI + Exploration Bonus
    # The exploration bonus is proportional to uncertainty (sigma)
    score = ei + lambda_explore * sigma
    
    # Ensure scores are finite
    score[~np.isfinite(score)] = 0.0
    
    return score
