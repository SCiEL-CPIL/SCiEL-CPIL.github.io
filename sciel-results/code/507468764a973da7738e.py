"""Rewritten acquisition for Bayesian optimization of 6d Hartmann."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_acq_start_time = None


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI) with adaptive exploration.
    
    PI scores candidates by the probability that they yield a value strictly better
    than f_best, providing robust exploitation. A small sigma-based bonus maintains
    early exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (n,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    global _acq_start_time
    if _acq_start_time is None:
        _acq_start_time = time.time()
    
    # Deadline guard: if we're running too long, return a simple fallback
    elapsed = time.time() - _acq_start_time
    if elapsed > 280:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        return np.where(np.isfinite(pi), pi, 0.0)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement (for minimization):
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Adaptive exploration bonus:
    # Early in optimization (high uncertainty), add a small sigma-based bonus
    # to encourage exploration. As the model becomes more certain, this bonus
    # naturally diminishes relative to the PI signal.
    # Scale by the mean sigma to normalize the bonus magnitude.
    sigma_mean = np.mean(sigma_safe)
    # Use a small weight (0.02) to keep PI dominant but allow some exploration
    exploration_bonus = 0.02 * sigma_safe
    
    # Combined score: PI is in [0,1], so the bonus should be small relative to that
    scores = pi + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
