"""Rewritten acquisition for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6d.
    
    Uses a Gaussian Process Upper Confidence Bound (UCB) strategy
    adapted for minimization. This is a structural change from the
    previous Hybrid EI approach. The score is defined as the negative
    mean plus a weighted uncertainty term, encouraging exploration
    of high-variance regions while exploiting low-mean regions.
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 300s limit
    
    n = len(mu)
    
    # Guard against non-positive sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # UCB for minimization:
    # We want to minimize f(x).
    # A lower bound on f(x) is mu(x) - beta * sigma(x).
    # We maximize the "optimistic" potential, which is equivalent to
    # minimizing the lower bound.
    # Score = -(mu(x) - beta * sigma(x)) = -mu(x) + beta * sigma(x)
    
    # Determine beta dynamically.
    # If the posterior is very uncertain (high sigma relative to f_best scale), explore more.
    # If the posterior is certain (low sigma), exploit more.
    
    # Scale of the problem
    f_scale = abs(f_best) + 1.0
    
    # Mean uncertainty
    mean_sigma = np.mean(sigma_safe)
    
    # If mean sigma is high compared to f_scale, we are in early exploration phase.
    # Use a higher beta.
    # If mean sigma is low, use a lower beta.
    
    # Normalize sigma by f_scale
    normalized_sigma = mean_sigma / f_scale
    
    # Beta ranges from ~0.5 (exploit) to ~2.0 (explore)
    # Linear interpolation based on normalized_sigma
    # If normalized_sigma > 1.0, beta = 2.0
    # If normalized_sigma < 0.1, beta = 0.5
    # Clamp normalized_sigma to [0.1, 1.0] for stable beta
    norm_sig_clamped = np.clip(normalized_sigma, 0.1, 1.0)
    
    # beta = 0.5 + 1.5 * norm_sig_clamped
    beta = 0.5 + 1.5 * norm_sig_clamped
    
    # Calculate scores
    # Higher score is better.
    # -mu favors low mu (good for minimization).
    # +beta*sigma favors high sigma (exploration).
    scores = -mu + beta * sigma_safe
    
    # Check deadline
    if time.time() > deadline:
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
