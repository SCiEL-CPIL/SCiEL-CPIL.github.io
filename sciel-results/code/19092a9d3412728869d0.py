"""Stochastic Expected Improvement acquisition with budget-aware exploration."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function.
    
    Combines standard Expected Improvement with a cost penalty that accounts
    for the stochastic nature of evaluation costs, while also incorporating
    an exploration term that scales with uncertainty to balance exploitation
    and exploration under a budget constraint.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (2000,)
        Deterministic cost estimates for each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware penalty: model cost uncertainty as log-normal with
    # constant log-space standard deviation beta.
    # The expected improvement per unit cost is:
    #   EI * E[1/C] where C ~ LogNormal(log(cost), beta)
    #   E[1/C] = exp(-log(cost) - beta^2/2)
    # So the cost-normalized EI is:
    #   (EI / cost) * exp(-beta^2/2)
    #
    # Since exp(-beta^2/2) is constant, the ranking is same as EI/cost.
    # To add a meaningful stochastic effect, we incorporate the cost
    # variance into the exploration-exploitation tradeoff.
    
    # Use a moderate beta for log-normal cost uncertainty
    beta = 0.15
    
    # Base cost-normalized EI
    cost_safe = np.maximum(cost, 1e-10)
    ei_cost = ei / cost_safe
    
    # Add an exploration bonus that is larger when uncertainty is high
    # relative to the expected improvement, encouraging exploration
    # in regions with high sigma but also accounting for cost.
    # The term sigma / cost encourages exploration in cheap, uncertain regions.
    exploration_weight = 0.3
    exploration = exploration_weight * sigma_safe / cost_safe
    
    # Combine: exploit where EI is high per unit cost, explore where
    # uncertainty is high per unit cost
    scores = ei_cost + exploration
    
    # Apply a soft penalty for very high costs to prefer cheaper evaluations
    # when EI is comparable
    high_cost_penalty = 0.1
    cost_penalty = high_cost_penalty * np.log1p(cost_safe)
    scores = scores - cost_penalty * (ei_cost < 0.1).astype(float)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
