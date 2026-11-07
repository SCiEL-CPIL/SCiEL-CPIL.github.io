import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s to be safe within 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Hybrid cost-aware acquisition function combining Expected Improvement (EI)
    and Lower Confidence Bound (LCB) for Bayesian optimization.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviation predictions.
        f_best: float, the lowest observed value.
        cost: (2000,) array of evaluation costs.
        
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next).
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in cost
    cost_safe = np.where(cost > 1e-6, cost, 1e-6)
    
    # 1. Calculate Standard Expected Improvement (EI)
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # Use vectorized norm.pdf and norm.cdf
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    # EI is theoretically non-negative for minimization when z is defined this way
    # (f_best - mu) is positive if mu < f_best, which is the interesting case.
    # If mu > f_best, EI is 0.
    ei = np.maximum(ei, 0.0)
    
    # 2. Calculate Lower Confidence Bound (LCB) score
    # For minimization, LCB = mu - beta * sigma
    # We want to maximize the "desirability", so we use -LCB = beta * sigma - mu
    beta = 2.0
    lcb_score = beta * sigma - mu
    
    # 3. Combine EI and LCB scores
    # Equal weighting
    hybrid_score = 0.5 * ei + 0.5 * lcb_score
    
    # 4. Cost-aware score
    score = hybrid_score / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
