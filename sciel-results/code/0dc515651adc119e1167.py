"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: evaluator kills after 330s. We track elapsed time for safety.
_START_TIME = time.time()
DEADLINE = _START_TIME + 315.0  # 15s safety margin


def _penalized_ei(mu, sigma, f_best):
    """
    Cost-aware Expected Improvement with a sigma-dependent penalty on the
    exploration term. The standard EI for minimization is:
        EI = (f_best - mu) * Phi(z) + sigma * phi(z),  z = (f_best - mu) / sigma
    Here we penalize the exploration term sigma*phi(z) by multiplying it by a
    factor that decreases with sigma, so high-uncertainty regions are suppressed
    in the sparse-data regime. This forces the search to prioritize high-confidence
    improvements over risky exploration.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Standard EI components
    cdf_part = improvement * norm.cdf(z)
    pdf_part = sigma_safe * norm.pdf(z)
    
    # Penalty on the exploration term: reduce sigma*pdf(z) when sigma is large
    # Use a soft penalty: scale pdf_part by exp(-sigma / scale)
    # scale controls how aggressively we suppress high-uncertainty regions
    scale = 0.05
    penalty = np.exp(-sigma_safe / scale)
    
    ei = cdf_part + penalty * pdf_part
    ei = np.maximum(ei, 0.0)
    return ei


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Griewank 2D.
    
    Uses Penalized Expected Improvement (PEI) that strictly penalizes the
    uncertainty term in the numerator using the standard deviation to prevent
    the acquisition function from being dominated by high-noise regions in the
    early, sparse data regime, thereby forcing the search to prioritize
    high-confidence improvements over risky exploration.
    """
    # Check deadline - if we're close to timeout, return a safe fallback
    if time.time() > DEADLINE:
        # Fallback: use penalized EI / cost
        pei = _penalized_ei(mu, sigma, f_best)
        scores = pei / np.maximum(cost, 1e-9)
        return scores

    # Compute Penalized Expected Improvement
    pei = _penalized_ei(mu, sigma, f_best)
    
    # Normalize by cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = pei / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
