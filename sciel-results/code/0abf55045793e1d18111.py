"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running too long, return quickly with a reasonable score
    # (In practice, the evaluator calls this function many times; we just need to be fast.)
    
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    # Avoid numerical issues with large z
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Also compute Probability of Improvement (PI) as a secondary signal
    pi = norm.cdf(z)
    
    # Cost-aware scoring: we want to balance improvement potential against cost.
    # Use a weighted combination:
    #   score = alpha * EI / cost + (1-alpha) * PI / cost
    # But actually, a more principled approach for cost-aware BO is:
    #   score = EI / cost  (or EI * exp(-beta * cost))
    #
    # Let's try: score = EI / (cost + small_offset) where small_offset prevents extreme values
    # and also add a small exploration term based on sigma to encourage exploring uncertain regions.
    
    cost_safe = np.maximum(cost, 1.0)
    
    # Base score: EI normalized by cost
    score_ei = ei / cost_safe
    
    # Exploration bonus: reward high sigma (uncertainty) regions, discounted by cost
    exploration = sigma_safe / cost_safe
    exploration = np.clip(exploration, 0, 1)
    
    # Combine: primarily EI/cost, with a small exploration component
    alpha = 0.9
    scores = alpha * score_ei + (1 - alpha) * exploration
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
