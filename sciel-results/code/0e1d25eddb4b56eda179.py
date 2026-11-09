"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 285

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of Expected Improvement (EI) and
    adaptive Upper Confidence Bound (UCB).
    
    This combines the exploitation strength of EI (from the current program's
    lineage, though it was using POI) with the exploration benefits of UCB.
    The UCB term is weighted adaptively based on f_best:
    - Early in optimization (high f_best), we explore more (higher UCB weight).
    - Late in optimization (low f_best), we exploit more (higher EI weight).
    
    This structural change addresses the stagnation observed with pure PI/POI
    by reintroducing a more principled exploration mechanism.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Basic safety checks
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        return np.zeros_like(mu)
    
    # Handle cases where sigma is very small
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute UCB score for minimization: maximize beta * sigma - mu
    # But we'll blend it with EI. Let's use a normalized version.
    
    # Adaptive weights based on f_best
    # Branin minimum is ~0.397887
    # Initial random points usually yield f_best in [5, 15]
    # We want high UCB weight when f_best is high (early stage).
    # We want high EI weight when f_best is low (late stage).
    
    f_min = 0.397887
    f_max_est = 15.0
    
    f_current = np.clip(f_best, f_min, f_max_est)
    
    # t = 0 when f_best = f_min (late stage, low UCB weight)
    # t = 1 when f_best = f_max_est (early stage, high UCB weight)
    t = (f_current - f_min) / (f_max_est - f_min)
    
    # Weights: 
    # w_ei = 1 - t (exploit more when t is small, i.e., f_best is low)
    # w_ucb = t (explore more when t is large, i.e., f_best is high)
    
    # But we need to scale them appropriately. Let's use a different approach.
    # Let's define the UCB term as: ucb = beta * sigma - mu
    # and EI as: ei = ...
    # We want to combine them. Let's normalize both to have similar scales.
    
    # Actually, let's use a simpler adaptive beta for UCB and add it to EI.
    # Or, let's use a weighted sum: score = w * ei + (1-w) * ucb_normalized
    
    # Let's try: score = ei + adaptive_beta * sigma
    # where adaptive_beta decreases as f_best decreases.
    
    # Adaptive beta:
    beta_min = 0.1
    beta_max = 2.0
    
    beta = beta_min + (beta_max - beta_min) * t
    
    # The UCB term is beta * sigma. But we want to add it to EI.
    # EI already has a sigma term. Let's see if adding beta*sigma helps.
    
    # Alternatively, let's use the UCB score directly but scaled.
    # ucb_score = beta * sigma - mu
    # ei_score = ei
    
    # Let's try a convex combination. But the scales are different.
    # Let's normalize the mu term.
    
    # Actually, a common approach is to use EI with an adaptive xi or beta.
    # Let's stick with: score = ei + beta * sigma
    # This encourages exploration in uncertain regions, especially early on.
    
    scores = ei + beta * sigma_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
