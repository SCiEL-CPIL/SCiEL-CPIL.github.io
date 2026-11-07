import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 290.0  # safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function for Bayesian optimization.
    
    Uses LCB = mu - beta * sigma, where beta is an adaptive exploration coefficient.
    The score is the negative LCB (so higher is better) normalized by cost.
    This explicitly trades off predicted improvement against uncertainty.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Clamp cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute dynamic exploration coefficient beta
    # Estimate the scale of the problem from the range of mu
    mu_range = np.ptp(mu)
    if mu_range < 1e-10:
        mu_range = 1.0
    
    # Adaptive beta: larger when sigma is large relative to mu range
    # This encourages more exploration when uncertainty is high relative to the scale
    sigma_median = np.median(sigma_safe)
    beta = 0.5 + 0.5 * (sigma_median / (mu_range + 1e-10))
    beta = np.clip(beta, 0.1, 2.0)
    
    # Lower Confidence Bound for minimization: mu - beta * sigma
    # We want to minimize this, so higher score = lower LCB
    LCB = mu - beta * sigma_safe
    
    # Score is -LCB (so higher is better) normalized by cost
    scores = -LCB / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if this took too long, return a simpler heuristic
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        # Fallback: simple LCB per cost with fixed beta
        simple_lcb = mu - 1.0 * sigma_safe
        simple_scores = -simple_lcb / cost_safe
        simple_scores = np.where(np.isfinite(simple_scores), simple_scores, 0.0)
        return simple_scores
    
    return scores
