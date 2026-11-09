import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0

def _time_check():
    """Check if we're close to the deadline."""
    return (time.time() - _START_TIME) > (_DEADLINE - _MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using quantile-based Probability of Improvement (POI).
    
    Computes P(f < f_best | x) = Phi((f_best - mu) / sigma) for each candidate.
    This ranks candidates by their likelihood of beating the incumbent,
    decoupling selection pressure from the absolute scale of the objective.
    
    We add a small exploration bonus based on quantile-normalized sigma to
    encourage visiting uncertain regions, but the primary signal is POI.
    """
    # Deadline guard: return a simple fallback if running out of time
    if _time_check():
        # Fallback: pure exploitation-like score to finish quickly
        return -mu

    # Ensure inputs are clean
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute z-scores: how many standard deviations f_best is above mu
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: P(f < f_best) = Phi(z)
    # For large positive z, POI ~ 1; for large negative z, POI ~ 0
    poi = norm.cdf(z)
    
    # Quantile-normalize sigma for exploration bonus
    # This ensures the exploration term is relative to typical uncertainty
    med_sigma = np.median(sigma)
    if med_sigma < 1e-12:
        med_sigma = 1e-12
    sigma_norm = sigma / med_sigma
    
    # Adaptive exploration weight:
    # Early on (high f_best), we want more exploration
    # Late (low f_best), we want more exploitation
    f_best_safe = max(f_best, 0.0)
    # Map f_best from [0, ~20] to [0, 1]
    ratio = np.log1p(f_best_safe) / np.log1p(5.0)
    ratio = np.clip(ratio, 0.0, 1.0)
    
    # Exploration weight: starts at 0.3, goes to 0.05
    # POI is in [0,1], so we scale the exploration term to be comparable
    eps = 0.05 + 0.25 * ratio
    
    # Combined score: primarily POI, with a small exploration bonus
    # Higher is better
    scores = poi + eps * sigma_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
