"""Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture in 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    
    # Deadline guard: if we are too close to the limit, return a simple score
    if time.time() - _start_time > 325:
        return -mu
    
    t = _call_count
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute Expected Hypervolume Improvement (EHI) approximation
    # EHI approximates the expected improvement in the hypervolume dominated by the best point
    # For a single-point improvement, this is related to EI but with a different weighting
    # that favors points that could significantly beat the current best
    
    # Use a weighted combination of EI and a variance-based exploration term
    # The key insight: in early stages, we want to explore more; in later stages, exploit more
    
    # Adaptive balance: start with more exploration, shift to exploitation
    # alpha controls the weight on exploration (sigma) vs exploitation (EI)
    alpha = 1.0 / (1.0 + 0.3 * t)
    
    # Exploration term: normalized sigma
    # We use sigma directly but scaled to be comparable to EI
    # EI has units of objective value, sigma also has units of objective value
    # So we can combine them directly
    
    # Additional structural change: use a "probabilistic regret" formulation
    # Instead of pure EI, use the probability of improvement weighted by expected improvement
    # P(improve) = Phi(z) where z = (f_best - mu) / sigma
    # This naturally down-weights points with low probability of improvement
    
    p_improve = norm.cdf(z)
    
    # Regret-weighted acquisition: emphasize points that are likely to improve
    # and have high expected improvement
    # The product p_improve * (f_best - mu) gives expected improvement contribution
    # But we also want to account for uncertainty in the improvement
    
    # Use a hybrid: standard EI plus a term that rewards high uncertainty
    # in regions where improvement is likely
    uncertainty_bonus = sigma_safe * p_improve
    
    # Combine EI and uncertainty bonus with adaptive weighting
    scores = ei + alpha * 0.5 * uncertainty_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
