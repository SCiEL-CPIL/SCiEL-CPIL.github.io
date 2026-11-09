import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_LIMIT = 285.0  # Safety margin below 300s

def acquisition(mu, sigma, f_best, cost):
    # Deadline check
    if time.time() - _START_TIME > _LIMIT:
        return np.zeros_like(mu)

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    # Clip z to prevent numerical underflow/overflow in pdf/cdf
    z = np.clip(z, -10, 10)

    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Cost-aware EI with refined exploitation bias
    # Stronger exploitation bias to converge faster in flat regions
    exploitation_bias = -0.15 * mu

    # Small uncertainty bonus: rewards moderate sigma (encourages exploration
    # in uncertain regions but penalizes very high uncertainty)
    # Using a Gaussian-like shape centered at sigma ~ 0.1 * range of mu
    mu_range = np.max(mu) - np.min(mu) + 1e-10
    target_sigma = 0.1 * mu_range
    uncertainty_bonus = 0.05 * target_sigma * np.exp(-0.5 * ((sigma - target_sigma) / (0.5 * target_sigma + 1e-10))**2)

    score = (ei + exploitation_bias + uncertainty_bonus) / cost

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
