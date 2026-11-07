"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function that balances expected improvement against cost.
    
    Uses a dynamic exploration-exploitation tradeoff:
    - Early on (low total cost consumed), favor exploration (higher uncertainty weight)
    - Later (high total cost consumed), favor exploitation (lower uncertainty weight)
    
    The acquisition score is: EI / (1 + beta * cost_consumed_ratio)
    where cost_consumed_ratio = total_cost / max_cost_budget
    
    Since we don't know the exact total cost consumed, we approximate it using
    the mean of the provided cost array as a proxy for the current evaluation cost,
    and use a fixed reference budget to create a dynamic penalty.
    
    Actually, re-reading: 'cost' has shape (2000,) and represents the cost of evaluating
    each of the 2000 candidates. The total cost consumed so far isn't directly given,
    but we can infer the budget is 30. We'll use a simpler approach:
    
    Use qEI with a cost-aware normalization that dynamically shifts based on
    how much budget has been used. Since we only have per-candidate costs, we'll
    use a heuristic: the mean cost of all candidates gives us a sense of the
    typical evaluation cost. We'll scale the EI by 1/sqrt(mean_cost) to penalize
    expensive regions, and add a small exploration term weighted by uncertainty.
    
    Better approach: Use the ratio of f_best to the typical scale to create
    a dynamic beta. In early iterations, f_best is large (far from minimum),
    so we explore more. As f_best decreases, we exploit more.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: normalize by cost, but add a dynamic exploration bonus
    # that decays as we get closer to the optimum
    # Use the mean cost as a baseline
    mean_cost = np.mean(cost)
    
    # Dynamic beta: starts high (exploration) when f_best is large,
    # decreases as f_best approaches 0
    # Scale f_best to [0, 1] assuming typical range
    f_scale = np.clip(f_best / 10.0, 0.0, 1.0)
    beta = 0.5 + 0.5 * f_scale  # beta in [0.5, 1.0]
    
    # Acquisition: EI / (cost * beta)
    # This penalizes expensive evaluations more when we're still exploring
    score = ei / (cost * beta)
    
    # Add a small exploration term for high-uncertainty regions
    # This helps escape local optima
    exploration_weight = 0.1 * f_scale
    score += exploration_weight * sigma_safe / (mean_cost + 1e-8)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
