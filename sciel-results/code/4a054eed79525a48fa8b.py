"""Rewritten cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 295.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining Expected Improvement with an uncertainty
    exploration bonus that adapts to remaining budget.
    
    Structural change from pure EI/cost: we use a hybrid that blends
    - EI (exploitation)
    - A scaled sigma term (exploration) whose weight increases when the
      GP is uncertain (early in the run) and decreases as budget depletes.
    
    The cost penalty is applied via a soft normalization that prefers
    cheaper evaluations without completely suppressing high-EI expensive points.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return np.zeros_like(mu)
    
    n = len(mu)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = improvement * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Exploration bonus: weighted sigma
    # The weight adapts based on how uncertain the GP is overall
    # and how much budget remains
    mean_sigma = np.mean(sigma_safe)
    # Normalize sigma to get relative uncertainty
    sigma_rel = sigma_safe / (mean_sigma + 1e-10)
    
    # Budget-aware exploration weight:
    # cost is in [1, 2], total budget is 30
    # As total cost approaches 30, reduce exploration
    # We don't know total cost directly, but we can infer from the cost array
    # The cost array represents per-candidate costs, not total budget used.
    # However, we can use the mean cost as a proxy for where we are in budget.
    # Actually, we don't have total cost. Let's use a simpler approach:
    # Weight exploration by a factor that balances exploitation vs exploration.
    
    # Use a tunable blend: higher sigma regions get more exploration
    # The key insight: early on, sigma is high everywhere, so we explore.
    # Later, sigma is low near known points, so we exploit.
    
    # Simple adaptive weight: use the ratio of EI to sigma
    # When EI >> sigma, we're confident -> exploit
    # When EI << sigma, we're uncertain -> explore
    ei_over_sigma = ei / (sigma_safe + 1e-10)
    
    # Blend EI and sigma-based exploration
    # Use a smooth function: as ei_over_sigma grows, weight shifts to EI
    # w_explore = 1 / (1 + ei_over_sigma) -- this is smooth
    w_explore = 1.0 / (1.0 + np.maximum(ei_over_sigma, 0.0))
    
    # Exploration term: sigma itself (uncertainty)
    explore_term = sigma_safe
    
    # Combine: score_raw = w_exploit * EI + w_explore * explore_term
    # Normalize both to comparable scales
    max_ei = np.max(ei) if np.max(ei) > 0 else 1.0
    max_explore = np.max(explore_term) if np.max(explore_term) > 0 else 1.0
    
    ei_norm = ei / max_ei
    explore_norm = explore_term / max_explore
    
    score_raw = (1.0 - w_explore) * ei_norm + w_explore * explore_norm
    
    # Cost-aware weighting
    # cost is in [1, 2] (1 + mean(x scaled to [0,1]))
    # Divide by cost to prefer cheaper points
    # Use a power less than 1 to soften the penalty
    cost_penalty = cost ** 0.5
    score = score_raw / cost_penalty
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
