"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track elapsed time to avoid timeout
_START_TIME = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0

def _time_check():
    """Check if we're close to the deadline."""
    return (time.time() - _START_TIME) > (_DEADLINE - _MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Hybrid EI/UCB acquisition for Bayesian optimization of Ackley 2D.
    
    Combines Expected Improvement (exploitation) with UCB (exploration)
    using an adaptive weight that shifts from exploration to exploitation
    as f_best approaches the minimum value of 0.
    
    mu, sigma: shape (2000,), posterior mean and std in objective units.
    f_best: lowest observed value.
    
    Returns 2000 finite scores, higher means evaluate next.
    """
    # Deadline guard: return a simple fallback if running out of time
    if _time_check():
        return 0.5 * sigma - mu
    
    n = len(mu)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Compute Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB with adaptive beta:
    # When f_best is high (early), explore more (higher beta)
    # When f_best is low (late), exploit more (lower beta)
    # Ackley 2D on [-5,5]^2 has values roughly in [0, ~20]
    # Typical initial f_best from uniform random starts might be ~1-5
    
    # Beta decays smoothly as f_best decreases
    # beta = 0.5 + 1.5 * clip(f_best / 5.0, 0, 1)
    # If f_best = 5, beta = 2.0 (explore)
    # If f_best = 0, beta = 0.5 (exploit)
    beta = 0.5 + 1.5 * np.clip(f_best / 5.0, 0.0, 1.0)
    
    UCB = -mu + beta * sigma
    
    # Adaptive weight for combining EI and UCB:
    # When f_best is close to 0, rely more on EI (exploitation)
    # When f_best is large, rely more on UCB (exploration)
    
    if f_best < 0.1:
        w_ucb = 0.2  # More exploitation
    elif f_best < 0.5:
        w_ucb = 0.4
    elif f_best < 2.0:
        w_ucb = 0.6
    else:
        w_ucb = 0.8  # More exploration early on
    
    # Combine: score = w_ucb * UCB + (1 - w_ucb) * EI
    scores = w_ucb * UCB + (1 - w_ucb) * EI
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
