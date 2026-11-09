import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware LCB acquisition function.
    
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
        # Fallback
        kappa = 2.0
        scores = (-mu + kappa * sigma) / cost
        return np.where(np.isfinite(scores), scores, -1e10)
    
    # LCB Acquisition:
    # We want to minimize f(x).
    # The lower confidence bound is L(x) = mu(x) - kappa * sigma(x).
    # We want to find x that minimizes L(x).
    # Acquisition score should be higher for better points.
    # So Score(x) = -L(x) / cost(x) = (-mu(x) + kappa * sigma(x)) / cost(x).
    
    # Tuning kappa:
    # A fixed kappa is requested for structural change.
    # 2.0 is a standard exploration value for LCB in BO.
    kappa = 2.0
    
    # Compute score
    scores = (-mu + kappa * sigma) / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
