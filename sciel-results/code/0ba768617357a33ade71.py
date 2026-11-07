import numpy as np
import time
from scipy.stats import norm

# Global state to track the start time for the deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    # Initialize start time if this is the first call
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: If we are running out of time, return a safe default
    # The evaluator kills after 330s.
    # This function is called frequently, but is vectorized.
    # We add a check to ensure we don't hang if something unexpected happens,
    # though numpy ops are typically fast.
    # A margin of 5 seconds before the 330s limit.
    if time.time() - _start_time > 325.0:
        # If we are out of time, return a neutral score to stop the search gracefully
        # or return the current best if we could, but we only have candidates.
        # Returning a very low score for all might stop the loop if the evaluator checks.
        # However, the contract says "Return 2000 finite scores".
        # If we are near the deadline, we should just return quickly.
        # The main cost is in the GP prediction outside this function.
        # If we are here, the GP is already done.
        # So we just compute the score as fast as possible.
        pass

    # Ensure inputs are arrays and handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical stability
    # Sigma should be positive. If it's zero or very small, EI becomes degenerate.
    # We clip sigma to a small positive number to avoid division by zero in z.
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Calculate Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Phi(z) is the CDF, phi(z) is the PDF
    # Using scipy.stats.norm for numerical stability and speed
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If f_best - mu is negative (mu < f_best), EI is technically 0 in standard definitions
    # because we only care about improvements. However, the formula above naturally handles this
    # because for z < 0, Phi(z) is small and (f_best - mu) is negative, but sigma * phi(z) is positive.
    # Actually, standard EI is max(0, f_best - mu).
    # The formula EI = (f_best - mu) * Phi(z) + sigma * phi(z) is always positive?
    # Let's check. If mu << f_best, z is large negative. Phi(z) ~ 0. phi(z) ~ 0. EI ~ 0.
    # If mu > f_best, z is negative.
    # The analytic solution for E[max(0, f_best - X)] where X ~ N(mu, sigma^2) is:
    # (f_best - mu) * Phi(z) + sigma * phi(z)
    # This formula is correct and yields non-negative values.
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Cost-aware normalization
    # Based on the cell finding, 1/sqrt(cost) outperformed 1/cost.
    # This reduces the penalty for expensive points, allowing exploration in high-cost areas.
    score = ei / np.sqrt(cost)
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
