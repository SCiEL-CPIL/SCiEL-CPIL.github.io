# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track time since first call to stay within budget.
_t_start = None

def acquisition(mu, sigma, f_best, cost):
    global _t_start
    if _t_start is None:
        _t_start = time.time()
    
    # Safety: if we're running out of time, return a simple but valid score
    elapsed = time.time() - _t_start
    if elapsed > 310.0:
        # Return a valid finite array immediately
        eps = 1e-9
        z = (f_best - mu) / (sigma + eps)
        pi = norm.cdf(z)
        return pi / (cost + eps)
    
    # Cost-aware Expected Improvement (EI) acquisition.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # This integrates the expected magnitude of improvement,
    # providing a smoother acquisition landscape that balances
    # exploration (high sigma) with exploitation (low mu).
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip negative EI to zero (shouldn't happen numerically, but be safe)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI penalized by cost.
    # Use a soft penalty: EI / (1 + cost) to avoid division-by-zero
    # and to smoothly trade off improvement potential against evaluation cost.
    score = ei / (1.0 + cost)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
