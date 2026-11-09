"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition combining expected improvement and uncertainty
    # with a deadline guard for safety.
    t0 = time.time()
    deadline = t0 + 250.0  # safety margin before 300s limit

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Cost-aware: divide by cost
    # But we also want to encourage exploration early on
    # Use a weighted combination: cost-aware EI + small bonus for high uncertainty
    # The uncertainty term helps escape local optima
    # Scale uncertainty appropriately
    # sigma is in objective units, so we can use it directly but scaled down
    
    # Primary: cost-aware expected improvement
    score = ei / cost
    
    # Secondary: exploration bonus based on uncertainty
    # This is small so it doesn't dominate, but helps in flat regions
    # Use a small coefficient to balance
    exploration_bonus = 0.01 * sigma / cost
    
    # Combine
    result = score + exploration_bonus
    
    # Ensure all values are finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        # Return current best result with safety margin
        pass
    
    return result
# EVOLVE-BLOCK-END
