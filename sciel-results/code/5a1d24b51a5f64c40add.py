# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    
    This approach selects points that maximize the expected improvement
    over the current best, penalized by the evaluation cost.
    
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    
    Score = EI(x) / cost(x)
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, improvement is f_best - f(x)
    # If mu > f_best, EI is 0 (or close to it). The formula above handles this naturally
    # because Phi(z) will be close to 0 and phi(z) small, but (f_best-mu) is negative.
    # Actually, standard EI for minimization is:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Let's verify signs.
    # If mu < f_best, z > 0. Phi(z) > 0.5. f_best - mu > 0. EI > 0.
    # If mu > f_best, z < 0. Phi(z) < 0.5. f_best - mu < 0.
    # The term (f_best - mu) * Phi(z) is negative.
    # The term sigma * phi(z) is positive.
    # Usually EI is defined as max(0, ...). The analytical formula yields 0 when improvement is not expected?
    # Let's check the limit. If mu >> f_best, z is large negative. Phi(z) -> 0. phi(z) -> 0. EI -> 0.
    # So the formula is correct and non-negative.
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: Higher EI per unit cost is better
    # cost is always >= 1 in this problem (1 + mean(x)), so no division by zero risk
    score = ei / np.maximum(cost, 1e-10)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
