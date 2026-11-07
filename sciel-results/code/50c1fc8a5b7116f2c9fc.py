"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time
    global _call_count
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are extremely close to the limit, return a safe fallback
    # In practice, this loop runs in < 1ms, so this is just a safety net.
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Fallback: return -mu (exploit current best) to ensure finite output
        return -np.asarray(mu)

    _call_count += 1
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Robustness: ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Structural Change: Use Lower Confidence Bound (LCB) for minimization
    # LCB(x) = mu(x) - kappa * sigma(x)
    # We want to minimize LCB, so we maximize -LCB(x) = -mu(x) + kappa * sigma(x)
    # This is a different acquisition function family than Expected Improvement.
    # We use a fixed kappa for now. A value of 1.0 is a common starting point.
    # Alternatively, we could use an adaptive kappa based on _call_count.
    # Let's try a constant kappa = 1.0 to see how it performs compared to EI.
    kappa = 1.0
    
    scores = -mu + kappa * sigma_safe
    
    # Ensure all values are finite to satisfy the contract
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
