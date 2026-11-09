"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with an adaptive UCB exploration bonus.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each point
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive UCB: exploration bonus that scales with uncertainty
    # The beta parameter adapts based on how far f_best is from the GP minimum
    # When f_best is close to min(mu), we reduce exploration; otherwise increase it
    min_mu = np.min(mu)
    # Normalize the gap: 0 means f_best is at GP min, 1 means f_best is far above
    gap = (f_best - min_mu) / (np.max(mu) - min_mu + 1e-9)
    gap = np.clip(gap, 0.0, 1.0)
    
    # Adaptive beta: more exploration when gap is large (early phase)
    beta = 0.5 + 1.5 * gap
    
    # UCB term
    ucb = mu + beta * sigma_safe
    
    # Blend EI and UCB: use EI primarily, add UCB exploration bonus
    # Normalize both to comparable scales
    ei_max = np.max(ei) + 1e-9
    ucb_max = np.max(np.abs(ucb)) + 1e-9
    
    # Blend with cost awareness: divide by cost
    # Higher cost points get penalized more
    cost_safe = np.maximum(cost, 1e-9)
    
    # Combine: weighted EI + UCB exploration, all divided by cost
    # The EI term drives exploitation, UCB drives exploration
    # We use a soft blend: when EI is high, rely more on EI; when EI is low, UCB helps
    ei_weight = 0.7
    ucb_weight = 0.3
    
    # Score = (ei_weight * EI + ucb_weight * |UCB improvement|) / cost
    # For UCB, we want to explore where ucb is high (i.e., mu + beta*sigma is high for minimization,
    # but since we're minimizing, we want low mu, so UCB for minimization is mu - beta*sigma)
    # Actually for minimization, UCB explores where mu - beta*sigma is low
    ucb_min = mu - beta * sigma_safe
    
    # Normalize ucb_min to be positive (higher is better for exploration)
    # Shift so that the minimum becomes 0
    ucb_shifted = np.maximum(ucb_min - np.min(ucb_min), 0.0)
    
    # Final score: blend of cost-normalized EI and cost-normalized UCB
    score = (ei_weight * ei + ucb_weight * ucb_shifted) / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
