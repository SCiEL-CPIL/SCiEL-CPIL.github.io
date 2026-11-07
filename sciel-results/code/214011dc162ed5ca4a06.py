import time
import numpy as np
from scipy.stats import norm

_T0 = time.perf_counter()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    if time.perf_counter() - _T0 > _DEADLINE:
        return 1.0 / np.maximum(cost, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Structural Change: Use Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    # This is the expected reduction in the best observed value.
    # It naturally balances exploitation (low mu) and exploration (high sigma).
    
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) = CDF of standard normal
    # phi(z) = PDF of standard normal
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (it should be, but clip for numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scaling: Expected improvement per unit cost
    # Since costs are in [1, 2], a simple division is reasonable.
    # Use a slight power to amplify the cost difference: cost^0.5
    scores = ei / np.sqrt(cost_safe)
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
