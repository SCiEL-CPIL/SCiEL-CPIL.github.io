"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = None
_DEADLINE_OFFSET = 325.0

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    
    # Safety margin: return simple EI if we are dangerously close to the 330s limit
    if elapsed > _DEADLINE_OFFSET:
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost

    # Structural Change: Knowledge Gradient (KG) Approximation via Hybrid PI + EI
    # The directive asks to decouple "discovery probability" from "value magnitude".
    # We use a weighted sum of Probability of Improvement (PI) and Expected Improvement (EI).
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # 1. Probability of Improvement (PI)
    # P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # 2. Expected Improvement (EI)
    # E[max(0, f_best - f(x))]
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to be comparable to PI?
    # EI has units of f, PI is probability [0,1].
    # To make them additive, we should scale EI.
    # A common scaling is to divide EI by a characteristic scale of the objective.
    # However, we don't know the scale. 
    # Alternatively, we can just weight them.
    # If f_best is small, EI is small.
    # Let's use a fixed weight for PI to ensure we always reward "chance of improvement".
    
    # Weight for PI (exploration/discovery)
    w_pi = 1.0
    # Weight for EI (exploitation/value)
    w_ei = 1.0
    
    # Combined Score
    # We want to maximize this score.
    # Both PI and EI are higher when the point is likely to be better.
    # We divide by cost to prefer cheaper points.
    
    score = (w_pi * pi + w_ei * ei) / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
