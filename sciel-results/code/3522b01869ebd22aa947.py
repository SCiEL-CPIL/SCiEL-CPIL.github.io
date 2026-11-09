"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of Expected Improvement (EI) and a cost-aware penalty,
    with a deadline guard to ensure we return within time limits.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Cost-aware scoring: divide by cost to favor cheaper evaluations
    # But also add a small exploration term based on uncertainty
    # Higher sigma means more uncertain, so we might want to explore
    
    # Use a weighted combination:
    # score = EI / cost + alpha * sigma / cost
    # where alpha controls exploration vs exploitation
    
    # Let's use a simpler approach: EI per unit cost
    # But boost slightly for high uncertainty regions
    
    # Cost-normalized EI
    score = ei / np.maximum(cost, 1e-8)
    
    # Add a small exploration bonus for high uncertainty
    # This helps escape local optima
    exploration_bonus = 0.1 * sigma_safe / np.maximum(cost, 1e-8)
    
    score = score + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check if we're close to the deadline
    if time.time() > deadline - 5.0:
        # Return early with what we have
        return score
    
    # Additional refinement: use a Pareto-like approach
    # Identify candidates that are good in both predicted value and cost
    # Score based on rank in (predicted min, cost) space
    
    # Compute a composite score that considers:
    # 1. How good the predicted minimum is (lower mu is better)
    # 2. How cheap the evaluation is (lower cost is better)
    # 3. How uncertain we are (higher sigma might be worth exploring)
    
    # Normalize each component
    mu_norm = (mu - np.min(mu)) / (np.max(mu) - np.min(mu) + 1e-8)
    cost_norm = (cost - np.min(cost)) / (np.max(cost) - np.min(cost) + 1e-8)
    sigma_norm = (sigma_safe - np.min(sigma_safe)) / (np.max(sigma_safe) - np.min(sigma_safe) + 1e-8)
    
    # Composite score: favor low mu, low cost, but allow some exploration
    composite = -0.5 * mu_norm - 0.3 * cost_norm + 0.2 * sigma_norm
    
    # Blend with the original EI-based score
    # Normalize both to [0, 1] range for blending
    score_min, score_max = np.min(score), np.max(score)
    score_norm = (score - score_min) / (score_max - score_min + 1e-8)
    
    comp_min, comp_max = np.min(composite), np.max(composite)
    comp_norm = (composite - comp_min) / (comp_max - comp_min + 1e-8)
    
    # Blend: 70% EI-based, 30% composite
    final_score = 0.7 * score_norm + 0.3 * comp_norm
    
    # Ensure finite
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
