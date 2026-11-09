import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware LCB acquisition function for Bayesian optimization (styblinski tang 2d).
    
    Uses a pure cost-normalized Lower Confidence Bound:
    score(x) = -(mu(x) - kappa * sigma(x)) / cost(x)
    
    Parameters
    ----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Deadline check
    def time_left():
        return 330.0 - (time.time() - t0) - 5.0
    
    if time_left() <= 0:
        # Fallback: simple LCB
        kappa = 2.0
        scores = (-mu + kappa * sigma) / cost
        return np.where(np.isfinite(scores), scores, -1e10)
    
    # Pure Cost-Normalized LCB
    # L(x) = mu(x) - kappa * sigma(x)
    # We want to minimize L, so score = -L / cost = (-mu + kappa*sigma) / cost
    
    # Adaptive kappa: balance exploration and exploitation
    # Start with a moderate kappa and adjust based on uncertainty
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    # Base kappa for exploration
    kappa_base = 1.5
    
    # Increase kappa when uncertainty is high relative to function scale
    uncertainty_ratio = sigma_max / (abs(f_best) + 1.0)
    kappa = kappa_base + 0.5 * np.clip(uncertainty_ratio, 0, 1.0)
    
    # Compute cost-normalized LCB score
    scores = (-mu + kappa * sigma) / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
