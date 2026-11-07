"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Uses the analytical EI formula for a GP posterior with known mean and std.
    EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    We divide by cost to make it cost-aware: score = EI / cost
    Higher score means better value per unit cost.
    """
    # Deadline guard
    if time.time() > _TIME_LIMIT:
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Guard against degenerate values
    cost = np.maximum(cost, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10.0, 10.0)
    
    # Compute Phi(z) and phi(z)
    from scipy.special import ndtr, ndtri
    # Actually, use scipy.stats for accuracy
    from scipy.stats import norm as _norm
    
    Phi = _norm.cdf(z)
    phi = _norm.pdf(z)
    
    # Expected Improvement
    # For minimization: EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # When f_best - mu > 0 (mu is worse than best), EI > 0
    # When f_best - mu <= 0 (mu is better than best), EI is small/zero
    
    delta = f_best - mu
    EI = delta * Phi + sigma * phi
    
    # Ensure EI is non-negative (numerical safety)
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware score: EI / cost
    scores = EI / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
