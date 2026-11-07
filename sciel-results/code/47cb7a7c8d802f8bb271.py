# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = time.time() + 285.0  # safety margin before the 300 s limit


def acquisition(mu, sigma, f_best, cost):
    t0 = time.time()
    if t0 > _DEADLINE:
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Defensive finiteness
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma = np.maximum(sigma, 1e-9)

    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # If f_best <= mu, EI is 0 (or very small)
    
    z = (f_best - mu) / sigma
    
    # Handle cases where z is very large or small to avoid numerical issues
    # Phi(z) -> 1 as z -> inf, Phi(z) -> 0 as z -> -inf
    # phi(z) is the standard normal PDF
    
    # For stability, we can use the standard formula
    # Note: norm.pdf(z) and norm.cdf(z) are vectorized
    
    phi_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # EI is defined as E[max(f_best - f(x), 0)]
    # If f(x) ~ N(mu, sigma^2), then E[max(f_best - f(x), 0)] = (f_best - mu)Phi(z) + sigma phi(z)
    # This formula works for all z, but if z is very negative, f_best - mu is negative, Phi(z) is small,
    # but the sum should be positive.
    # Actually, if mu > f_best, z is negative.
    # Let's ensure we don't get negative EI due to floating point errors.
    
    ei = (f_best - mu) * cdf_z + sigma * phi_z
    
    # EI should be non-negative. Clamp to 0.
    ei = np.maximum(ei, 0.0)
    
    # Cost Adjustment
    # We want to maximize EI / Cost
    # Cost is in [1, 2]
    # Dividing by cost penalizes expensive points.
    
    score = ei / cost
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
