"""Cost-aware acquisition for three-hump-camel-2d BO with probabilistic regret minimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite acquisition scores (higher = evaluate next).
    
    Uses a probabilistic framework to maximize the probability of achieving 
    a target regret reduction per unit cost. This models the trade-off between 
    expected improvement and evaluation cost more explicitly than simple EI/cost.
    """
    t0 = time.time()
    
    # Ensure sigma is safe for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (standard)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probabilistic regret minimization:
    # We model the probability of achieving a target regret reduction R per unit cost.
    # Let R = f_best - mu (improvement). We want to maximize P(R > threshold) / cost.
    # Using a smooth epigraph formulation: P(R > threshold) ≈ Phi((R - threshold) / sigma)
    # where threshold is adaptive based on current best.
    
    # Adaptive threshold: scale with uncertainty
    # threshold = 0.1 * f_best (small target for early exploitation)
    # But we want to balance exploration and exploitation.
    
    # Smooth surrogate for the probability of improvement:
    # Instead of hard threshold, use a soft probability via CDF
    # P(improvement > delta) = Phi((f_best - mu - delta) / sigma)
    
    # Choose delta adaptively: small delta for exploitation, larger for exploration
    # delta = 0.01 * f_best + 0.01 (small target)
    delta = 0.01 * np.abs(f_best) + 0.01
    
    # Probability of achieving improvement greater than delta
    z_prob = (f_best - mu - delta) / sigma_safe
    prob_improve = norm.cdf(z_prob)
    
    # Cost-aware score: prob_improve / cost
    # This directly models "probability of meaningful improvement per unit cost"
    cost_safe = np.maximum(cost, 1e-9)
    scores = prob_improve / cost_safe
    
    # Add a small exploration bonus for high uncertainty regions
    # to avoid premature convergence
    exploration_bonus = 0.05 * sigma_safe / cost_safe
    
    scores = scores + exploration_bonus
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard
    if time.time() - t0 > 280.0:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
