import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Quantile of Improvement (QI) acquisition function.
    
    QI = EI / PI when PI > threshold, else 0.
    This represents the conditional expected improvement given that an improvement occurs.
    It balances exploration and exploitation by focusing on regions with both
    high probability and high magnitude of improvement.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores (higher is better).
    """
    # Deadline guard
    t_start = time.time()
    deadline = 300.0
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    pi_scores = norm.cdf(z)
    
    # Expected Improvement: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    phi_z = norm.pdf(z)
    ei = (f_best - mu) * pi_scores + sigma_safe * phi_z
    
    # Quantile of Improvement: QI = EI / PI when PI > 0, else 0
    # This represents the expected improvement given that an improvement occurs
    q_threshold = 0.01  # small threshold to avoid division by near-zero
    pi_safe = np.where(pi_scores > q_threshold, pi_scores, np.inf)
    qi = np.where(pi_scores > q_threshold, ei / pi_safe, 0.0)
    
    # Cost-aware score: Maximize QI per unit cost
    score = qi / cost
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check deadline
    t_elapsed = time.time() - t_start
    if t_elapsed > deadline - 5.0:
        # Safety margin: return current result
        return score
    
    return score
