"""Acquisition function for Bayesian optimization of ackley 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0

def _time_check():
    """Check if we're close to the deadline."""
    return (time.time() - _START_TIME) > (_DEADLINE - _MARGIN)

def acquisition(mu, sigma, f_best):
    """
    UCB acquisition with dynamic exploration weight.
    
    The exploration weight beta is scaled by the current best value f_best
    and the quantile-normalized sigma, promoting aggressive early exploration
    of high-variance regions to escape local minima in Ackley, transitioning
    to exploitation as f_best approaches 0.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases: if sigma is all zeros or very small, just return -mu
    if np.all(sigma < 1e-10):
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        return scores
    
    # Quantile-normalize sigma to [0, 1] for stable exploration weighting
    # This makes the exploration weight robust to scale changes in the GP posterior
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    if sigma_max - sigma_min < 1e-10:
        sigma_norm = np.zeros_like(sigma)
    else:
        sigma_norm = (sigma - sigma_min) / (sigma_max - sigma_min)
    
    # Dynamic exploration weight:
    # - Start with high exploration when f_best is far from optimum (0)
    # - Decay exploration as f_best approaches 0
    # - Scale by the quantile-normalized sigma to emphasize high-variance regions
    
    # Base exploration weight that decays with progress
    # When f_best is large (early), beta is large; when f_best is small (late), beta is small
    # Use a sigmoid-like decay: beta = beta_max * exp(-f_best / scale)
    # But we want more aggressive early exploration, so let's use:
    # beta = beta_max * (1 + f_best) / (1 + f_best) ... no, that's constant
    
    # Better: beta proportional to the remaining gap to optimum
    # Since the minimum is 0, the gap is f_best (assuming f_best >= 0 for Ackley)
    # But f_best could be negative in some cases, so clip it
    f_best_clipped = max(f_best, 0.0)
    
    # Exploration weight: starts high, decays as we get closer to optimum
    # Use a power law or exponential decay
    # beta = beta_max * exp(-alpha * f_best_clipped)
    # With alpha = 1.0, beta decays from beta_max at f_best=0 to beta_max*exp(-1) at f_best=1
    
    # Actually, for Ackley, the function values are typically in [0, ~30]
    # So we want aggressive exploration when f_best is large
    # Let's use: beta = beta_max * (f_best_clipped + 1) / (f_best_clipped + 1)
    # That's still constant. Let's think differently.
    
    # The key insight: early on, we want to explore high-variance regions aggressively.
    # The UCB score is: mu + beta * sigma
    # We want beta to be large when:
    # 1. f_best is far from 0 (early in optimization)
    # 2. sigma is high (uncertain regions)
    
    # For (1), we can scale beta by a function of f_best
    # For (2), we already have sigma in the UCB formula
    
    # Let's use: beta = beta_base * (1 + f_best_clipped)
    # This increases exploration as f_best increases (i.e., when we're worse)
    # But we want the opposite: more exploration when we're worse means f_best is larger
    # Wait, if f_best is large, we're worse, so we should explore more. That's correct.
    
    # But we also want to transition to exploitation as f_best approaches 0.
    # So beta should decrease as f_best approaches 0.
    
    # Let's use: beta = beta_max * exp(-f_best_clipped / tau)
    # With tau = 1.0, this gives:
    # f_best=0: beta = beta_max
    # f_best=1: beta = beta_max * 0.37
    # f_best=5: beta = beta_max * 0.0067
    
    # This seems too aggressive in decay. Let's use tau = 5.0:
    # f_best=0: beta = beta_max
    # f_best=5: beta = beta_max * 0.37
    # f_best=10: beta = beta_max * 0.135
    
    # For Ackley 2D, typical f_best values might be in [0, 10]
    # So tau = 5.0 seems reasonable
    
    beta_max = 3.0  # Maximum exploration weight
    tau = 5.0  # Decay constant
    
    beta = beta_max * np.exp(-f_best_clipped / tau)
    
    # Compute UCB score: mu + beta * sigma
    # We want to minimize, so higher score = better candidate
    # UCB for minimization: -mu + beta * sigma (we negate mu since lower is better)
    # Or equivalently: -(mu - beta * sigma)
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
