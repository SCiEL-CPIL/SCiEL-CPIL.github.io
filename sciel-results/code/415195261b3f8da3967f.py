# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 330.0
_SAFETY_MARGIN = 5.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Strategy: Pure Cost-Adjusted Expected Improvement (EI).
    We use a "virtual best" threshold slightly above f_best to encourage
    exploration of regions that are not yet fully explored but have high
    uncertainty. The cost is handled by dividing the EI by the cost,
    effectively maximizing improvement per unit cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of GP posterior in objective units.
    sigma : ndarray, shape (2000,)
        Standard deviation of GP posterior in objective units.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - _SAFETY_MARGIN:
        mu = np.asarray(mu, dtype=float)
        cost = np.asarray(cost, dtype=float)
        mu = np.where(np.isfinite(mu), mu, f_best)
        cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        return -mu / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma >= 0), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # --- Expected Improvement (EI) for minimization ---
    # Standard EI uses f_best. 
    # To encourage exploration in multimodal functions like Rastrigin,
    # we can use a threshold slightly worse than f_best.
    # However, standard EI is often robust. Let's stick to standard EI
    # but ensure the cost normalization is effective.
    
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in exp/cdf
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # --- Cost-Awareness ---
    # We want to maximize EI per unit cost.
    # cost is in [1, 2].
    # Dividing by cost directly penalizes expensive points.
    # However, simple division might not be enough if EI values vary widely.
    # Let's use a soft cost penalty: EI / (cost ** alpha)
    # alpha = 1.0 is standard.
    
    scores = ei / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
