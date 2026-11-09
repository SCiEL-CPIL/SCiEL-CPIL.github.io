import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Returns 2000 finite scores (higher = evaluate next).
    Uses Cost-Adjusted Probability of Improvement (PI / Cost).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    # Using a very small epsilon to avoid division by zero in z calculation
    # PI is well-behaved even if sigma is 0 (z becomes +/- inf), 
    # but norm.cdf(inf) = 1, norm.cdf(-inf) = 0.
    # If sigma is 0, z = (f_best - mu) / 0.
    # If mu < f_best, z = +inf, PI = 1.
    # If mu > f_best, z = -inf, PI = 0.
    # If mu == f_best, z = 0/0 = NaN.
    # So we must handle sigma=0.
    
    # We want to avoid NaNs.
    # If sigma is extremely small, we can treat it as 0.
    # Let's use a threshold.
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Probability of Improvement
    # PI = P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-12)
    
    # Score = PI / Cost
    score = pi / cost_safe
    
    # Handle non-finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
