# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Expected Improvement (CA-EI) with Analytic Gaussian Integration.
    
    Strategy:
    1. Use Expected Improvement (EI) instead of UCB. EI is theoretically grounded
       in the probability of improvement and is more stable in the presence of 
       noise and multi-modality. It naturally balances exploitation (low mu) and 
       exploration (high sigma) via the Gaussian tail.
    2. Adjust for cost by dividing the expected improvement by the evaluation cost.
       This favors points that are both likely to improve the best found value 
       AND cheap to evaluate (center of the domain).
    3. Use the analytic formula for EI under a Gaussian posterior to ensure 
       numerical stability and speed.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Predictive mean from the GP.
    sigma : array-like, shape (2000,)
        Predictive standard deviation from the GP.
    f_best : float
        Lowest observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Deadline guard: Return zeros if we are close to the time limit
    if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
        return np.zeros_like(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    # Compute the standard normal variable
    z = (f_best - mu) / sigma
    
    # Calculate Phi(z) and phi(z)
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # Compute EI
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Cost-Adjusted EI:
    # We want to maximize improvement per unit cost.
    # scores = EI / cost
    scores = ei / cost
    
    # Ensure scores are finite and non-negative (EI should be non-negative, but numerical errors can occur)
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
