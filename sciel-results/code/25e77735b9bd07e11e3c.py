"""Rewritten program for cost-aware acquisition (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute constants for speed
_EPS = 1e-12


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    We use a modified Expected Improvement (EI) that accounts for the cost of evaluation.
    The cost is given as an array of the same shape as mu/sigma.
    cost = 1 + mean(x_scaled) where x_scaled is in [0,1].
    
    To balance exploration and cost, we compute EI per unit cost.
    However, simple EI/cost can be too aggressive in early stages.
    
    A robust approach is to use a "discounted" EI or to add a small exploration bonus 
    that decays with cost, or simply use EI / (cost ** alpha).
    
    Let's stick with the standard cost-aware EI: EI / cost.
    But we must ensure numerical stability.
    
    z = (f_best - mu) / sigma
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    Score = EI / cost
    
    We handle sigma=0 by treating it as a certain improvement if mu < f_best.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, _EPS)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For numerical stability with large |z|:
    # If z is very large positive, Phi(z) ~ 1, phi(z) ~ 0 => EI ~ f_best - mu
    # If z is very large negative, Phi(z) ~ 0, phi(z) ~ 0 => EI ~ 0
    
    # Standard calculation
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical artifacts can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI divided by cost
    # Cost is at least 1.0
    score = ei / cost
    
    # Return finite scores
    # Replace any non-finite values with 0
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
