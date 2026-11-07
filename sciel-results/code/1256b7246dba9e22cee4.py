"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition: sublinear cost penalty + exploration bonus
    # 1/sqrt(cost) allows exploration of high-cost regions that may contain the minimum
    # sigma bonus prevents premature convergence by rewarding uncertain regions
    
    # Safe sigma floor to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Sublinear cost penalty: 1/sqrt(cost) instead of 1/cost
    # This is less aggressive in penalizing high-cost regions
    cost_penalty = 1.0 / np.sqrt(np.maximum(cost, 1e-8))
    
    # Exploration bonus: reward high uncertainty
    # Scaled to be meaningful relative to EI
    exploration_bonus = 0.5 * sigma_safe / (np.abs(f_best) + 1.0)
    
    # Combine: EI with sublinear cost penalty + exploration bonus
    score = ei * cost_penalty + exploration_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
