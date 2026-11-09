"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Weighted Expected Improvement (EI).
    
    EI measures the expected amount of improvement over f_best.
    Dividing by cost prioritizes evaluations that yield high expected improvement
    per unit cost.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best observed value so far.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores. Higher is better.
    """
    start_time = time.time()
    
    # Ensure arrays are float
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Prevent division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Calculate Z-score for improvement
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0, high probability of improvement.
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu > f_best, z is negative.
    # Phi(z) is small, (f_best - mu) is negative.
    # The term (f_best - mu) * Phi(z) will be negative.
    # However, EI is usually defined as max(0, f_best - f(x)).
    # The formula above yields negative values when mu > f_best + sigma*phi(z)/Phi(z) approx.
    # To be safe and strictly non-negative, we can clip or use the standard formula.
    # Standard EI formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is correct for the expectation of max(0, f_best - Y) where Y ~ N(mu, sigma^2).
    # Let's verify: If mu is very large, z is large negative. Phi(z) ~ 0. (f_best-mu) is negative.
    # The term (f_best-mu)*Phi(z) is small negative. sigma*phi(z) is small positive.
    # Actually, for large negative z, phi(z) decays faster than Phi(z)? No.
    # Let's just compute it and clip to 0 to be safe, as EI should be non-negative.
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # EI should be non-negative. Numerical errors might make it slightly negative.
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    # We want to maximize EI per unit cost.
    # Cost is in [1, 2].
    # Using 1/cost is a standard way to penalize high cost.
    # To avoid division by zero or extreme values, we can smooth it.
    # But cost is >= 1, so 1/cost is in [0.5, 1].
    
    scores = ei / cost_safe
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 280:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
