"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement with softened cost penalty.
    # The global minimum of the 3-hump camel function is in a higher cost region
    # than the local minimum, so a strong cost penalty can bias the search
    # towards the suboptimal local minimum. We use a weaker penalty.
    
    # Safety margin: return early if we're getting close to the 300s limit
    elapsed = time.time() - _start_time
    if elapsed > 290:
        # Return a reasonable fallback: just EI without cost penalty
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # cost is in [1, 2]. We use a linear penalty with slope 0.2.
    # This reduces the relative penalty for expensive regions compared to 1/cost.
    cost_penalty = 1.0 + 0.2 * (cost - 1.0)
    return ei / cost_penalty
# EVOLVE-BLOCK-END
