import numpy as np
from scipy.stats import norm
import time

# Global variable to track start time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are too close to the limit, return a simple score
    # 330s total limit. Let's say we want to be done by 325s.
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        # Return a simple exploitation score: -mu
        # This ensures we pick the best current point if we are running out of time
        return -np.asarray(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Knowledge Gradient (KG) approximation
    # KG(x) = E[max(0, f_best - f(x))] - E[max(0, f_best - E[f(x)])]
    # For a normal distribution N(mu, sigma^2):
    # E[max(0, f_best - f)] = sigma * [phi(z) + z * Phi(z)] where z = (f_best - mu) / sigma
    # E[max(0, f_best - mu)] = max(0, f_best - mu)
    
    z = (f_best - mu) / sigma_safe
    
    # phi(z) = standard normal PDF
    phi = norm.pdf(z)
    # Phi(z) = standard normal CDF
    Phi = norm.cdf(z)
    
    # E[max(0, f_best - f)] = sigma * (phi(z) + z * Phi(z))
    # Note: for z < 0, Phi(z) is small but positive, and phi(z) is positive
    # For z > 0, this is the standard expected improvement formula
    
    # Actually, let's be more careful:
    # E[max(0, f_best - f)] where f ~ N(mu, sigma^2)
    # = sigma * [phi(z) + z * Phi(z)]  where z = (f_best - mu)/sigma
    # This is correct for all z.
    
    expected_improvement = sigma_safe * (phi + z * Phi)
    
    # E[max(0, f_best - E[f])] = max(0, f_best - mu)
    expected_improvement_mean = np.maximum(0.0, f_best - mu)
    
    # KG = E[max(0, f_best - f)] - E[max(0, f_best - E[f])]
    kg = expected_improvement - expected_improvement_mean
    
    # The KG is always non-negative. Higher is better.
    # However, KG tends to be small for points far from f_best.
    # To improve exploration in multi-modal landscapes, we can add a small
    # exploration bonus based on uncertainty, but weighted to not dominate.
    
    # Add a small exploration bonus: this helps explore high-uncertainty regions
    # while still favoring points likely to improve.
    # The bonus is proportional to sigma but scaled down.
    exploration_bonus = 0.1 * sigma_safe
    
    scores = kg + exploration_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
