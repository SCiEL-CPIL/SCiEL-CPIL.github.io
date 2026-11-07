"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s kill


def _within_deadline():
    return (time.time() - _start_time) < (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Bayesian model evidence acquisition: expected negative log-likelihood.
    
    Score = -((f_best - mu) / sigma) - log(sigma)
    
    This is the expected log-likelihood of observing f_best given the GP posterior,
    which naturally balances exploitation (low mu) against uncertainty penalty (log sigma).
    Higher score = more likely to be the next best evaluation.
    
    We normalize by cost to account for the cost-awareness requirement.
    """
    if not _within_deadline():
        # Fallback: pure exploitation weighted by cost
        return -mu / (cost + 1e-9)
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected negative log-likelihood of observing f_best
    # NLL = (f_best - mu)^2 / (2*sigma^2) + 0.5*log(2*pi*sigma^2)
    # But the directive suggests: -((f_best - mu) / sigma) - log(sigma)
    # This is a linear approximation that captures the key tradeoff.
    
    # Compute the evidence-based score
    # When mu < f_best, (f_best - mu) > 0, so -(f_best-mu)/sigma is negative (good, low cost)
    # When sigma is large, log(sigma) is large, so -log(sigma) is negative (penalty for uncertainty)
    # Wait - we want HIGHER score = better. Let me reconsider.
    
    # The expected log-likelihood of the best observation is:
    # E[log p(f_best | mu, sigma)] ~ (f_best - mu)/sigma - log(sigma) + const
    # Higher expected log-likelihood = more likely this point leads to improvement
    # So score = (f_best - mu) / sigma - log(sigma)
    # When mu is low (good), (f_best - mu) is large positive -> high score
    # When sigma is high, log(sigma) is large -> reduces score (uncertainty penalty)
    
    # But we need to be careful: if mu >> f_best, then (f_best - mu) is negative,
    # which correctly gives a low/negative score (this point is already worse than best).
    
    score = (f_best - mu) / sigma_safe - np.log(sigma_safe)
    
    # Cost normalization: divide by cost to get per-unit-cost value
    # Lower cost = higher score (more efficient)
    score = score / (cost + eps)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
