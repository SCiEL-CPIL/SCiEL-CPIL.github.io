"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track global start time for deadline safety
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to spare before 330s kill

# Constants
MIN_POSSIBLE = -0.8
MAX_POSSIBLE = 1.0  # Approx max of cosine8

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    
    Structural Change: 
    Replaces the UCB/PoI blend with Cost-Aware Expected Improvement (cEI).
    EI is theoretically grounded in maximizing the expected reduction in regret.
    By dividing EI by the evaluation cost, we explicitly optimize for 
    regret reduction per unit cost. This is more principled than ad-hoc 
    blending of UCB terms and handles the trade-off between finding a 
    slightly better point that is cheap vs. a much better point that is 
    expensive more effectively.
    
    Additionally, we apply a small jitter to sigma to avoid division by zero 
    and ensure numerical stability.
    """
    # Deadline guard
    if time.time() - _START_TIME > 330.0 - _DEADLINE_MARGIN:
        return np.zeros_like(mu)

    n = len(mu)
    
    # Numerical safety
    # Ensure sigma is strictly positive for EI calculation
    sigma_safe = np.maximum(sigma, 1e-8)
    # Cost is in [1, 2], ensure no zeros
    cost_safe = np.maximum(cost, 1.0)
    
    # Calculate Expected Improvement for minimization
    # Improvement = f_best - f(x)
    # E[I] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Clip z to avoid extreme values in Phi/phi
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -20, 20)
    
    # Phi(z) = norm.cdf(z)
    # phi(z) = norm.pdf(z)
    phi_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu > f_best, z is negative. (f_best - mu) is negative.
    # EI should be 0 if we are certain f(x) > f_best.
    # The formula above handles this naturally as (f_best - mu) becomes negative 
    # and Phi(z) becomes small, but technically EI can be slightly negative 
    # due to floating point or if we don't clamp. 
    # Standard EI is max(0, E[I]).
    
    ei = (f_best - mu) * phi_z + sigma_safe * pdf_z
    
    # Clamp EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    # We want to maximize the score.
    # Score = EI / Cost
    score = ei / cost_safe
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
