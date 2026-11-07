"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_bo_start_time = None

def acquisition(mu, sigma, f_best):
    global _bo_start_time
    
    # Initialize start time on first call
    if _bo_start_time is None:
        _bo_start_time = time.time()
        
    # Deadline Guard: 330s limit, use 310s as safety margin
    if time.time() - _bo_start_time > 310:
        # If we are close to timeout, prioritize speed and exploitation
        # to ensure the loop finishes.
        return -mu

    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and ensure positivity
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB score
    # We want a balance. The current program used adaptive beta.
    # The inspiration 1 (score 1.53) used fixed beta=1.0 UCB.
    # The current program (score 2.19) uses adaptive beta UCB.
    # Let's try a hybrid: EI + adaptive UCB.
    # Or perhaps just refine the adaptive UCB.
    
    # Let's look at the scores:
    # EI alone (insp 3): 1.0
    # UCB beta=1 (insp 1): 1.53
    # Hybrid EI+UCB (insp 2): 1.11
    # Adaptive UCB (current): 2.19
    
    # The adaptive UCB is currently the best.
    # Let's refine the adaptive beta strategy.
    # Current: beta = 0.5 + 2.5 * min(1.0, f_best / 10.0)
    # This means:
    # f_best = 0 -> beta = 0.5
    # f_best = 10 -> beta = 3.0
    # f_best > 10 -> beta = 3.0
    
    # Ackley 2D range is roughly [0, 30].
    # Early on, f_best is high (e.g., 20-30). Beta should be high to explore.
    # Later, f_best drops. Beta should drop to exploit.
    
    # Maybe the transition is too slow or the range is off?
    # Let's try a smoother transition or different bounds.
    # Also, consider adding a small EI component?
    # The current program is pure UCB.
    
    # Let's try: Score = -mu + beta * sigma
    # But maybe beta should depend on the number of evaluations?
    # We don't have the count directly, but f_best is a proxy.
    
    # Let's try to combine EI and UCB more effectively.
    # Score = EI + gamma * (UCB - EI)? No, that's complex.
    
    # Let's stick to the UCB family as it's performing best.
    # Refinement:
    # 1. Ensure beta is not too large when f_best is very small.
    # 2. Ensure beta is large enough when f_best is large.
    
    # Let's try a sigmoid-like transition for beta.
    # beta_min = 0.5, beta_max = 3.0
    # center = 5.0 (midpoint of interesting f_best range)
    # width = 5.0
    
    # beta = beta_min + (beta_max - beta_min) / (1 + exp((f_best - center) / width))
    # If f_best = 0: exp(-1) ~ 0.37, denom ~ 1.37, term ~ 0.73, beta ~ 0.5 + 2.5*0.73 = 2.3?
    # Wait, if f_best is low, we want low beta (exploit).
    # So as f_best -> 0, beta -> beta_min.
    # As f_best -> inf, beta -> beta_max.
    
    # Formula: beta = beta_min + (beta_max - beta_min) / (1 + exp(-(f_best - center) / width))
    # If f_best = 0, center=5, width=5: exp(1) ~ 2.7, denom ~ 3.7, term ~ 0.27, beta ~ 0.5 + 2.5*0.27 = 1.17
    # If f_best = 10, center=5, width=5: exp(0) = 1, denom ~ 2, term ~ 0.5, beta ~ 0.5 + 2.5*0.5 = 1.75
    # If f_best = 20, center=5, width=5: exp(3) ~ 20, denom ~ 21, term ~ 0.95, beta ~ 0.5 + 2.5*0.95 = 2.87
    
    # This looks reasonable.
    
    beta_min = 0.5
    beta_max = 3.0
    center = 5.0
    width = 5.0
    
    # Calculate adaptive beta
    # Use np.exp with clipping to avoid overflow
    exponent = np.clip((f_best - center) / width, -50, 50)
    beta = beta_min + (beta_max - beta_min) / (1 + np.exp(-exponent))
    
    # Score = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure finite values
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
