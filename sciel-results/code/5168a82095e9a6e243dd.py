"""Acquisition function for Bayesian optimization of a 2D RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # Safety margin

def _time_remaining():
    return _DEADLINE - time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling for minimization.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        GP posterior std at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising points.
    """
    # Check deadline
    if _time_remaining() < 0.5:
        # Fallback: deterministic EI or just -mu
        # If time is up, use a simple heuristic
        return -mu

    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: Draw a sample from the posterior
    # y ~ N(mu, sigma^2)
    sample = mu + sigma_safe * np.random.randn(len(mu))
    
    # We want to minimize the objective function.
    # The evaluator picks the candidate with the HIGHEST score.
    # Therefore, we return the negative of the sampled value.
    # Lower sampled value -> Higher score.
    scores = -sample
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
