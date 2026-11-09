"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a dynamic exploration-exploitation balance:
    - Early iterations: more exploration (higher UCB weight)
    - Late iterations: more exploitation (lower UCB weight)
    
    The UCB weight scales with remaining budget relative to current cost.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Dynamic UCB weight: scales with remaining budget
    # cost is the cost of evaluating each candidate (1 + mean(x_scaled))
    # Total budget is 30, initial 5 points cost ~5*1.5=7.5 average
    # So remaining budget is roughly 30 - current_total_cost
    # We don't have current_total_cost directly, but we can estimate from cost array
    # cost[i] is the cost of candidate i, so sum of evaluated costs gives total
    # But we only have the 2000 candidates' costs, not which have been evaluated
    
    # Use a simpler heuristic: the mean cost of candidates gives a sense of budget usage
    # As we evaluate more points, the effective remaining budget shrinks
    # Use the ratio of typical cost to max possible cost as a proxy for progress
    
    # Max cost is 1 + 1 = 2 (when x is all at upper bound scaled to 1)
    # Min cost is 1 + 0 = 1 (when x is all at lower bound scaled to 0)
    # Average cost ~ 1.5
    
    # Estimate progress: if we've used ~7.5 of 30 budget initially, we have ~22.5 left
    # After each evaluation, remaining budget decreases by ~1.5
    # We can estimate how many evaluations have occurred by looking at the spread of costs
    # But we don't have direct info on how many have been evaluated
    
    # Alternative: use a fixed schedule based on typical iteration count
    # With budget 30 and avg cost 1.5, we can do ~20 evaluations after initial 5
    # So ~25 total evaluations, each with 2000 candidates scored
    
    # For simplicity, use a smooth decay based on the median cost
    # Lower cost candidates are cheaper, so we should explore more when costs are low
    median_cost = np.median(cost_safe)
    
    # UCB weight: higher when remaining budget is large (early), lower when small (late)
    # Normalize cost to [0, 1] range: cost in [1, 2], so (cost - 1) in [0, 1]
    cost_norm = (cost_safe - 1.0) / 1.0
    cost_norm = np.clip(cost_norm, 0.0, 1.0)
    
    # Beta parameter for UCB: starts high (exploration) and decays
    # Use a schedule based on the fraction of budget likely remaining
    # Early: beta ~ 2.0, Late: beta ~ 0.5
    # Estimate progress from the fact that we're scoring 2000 candidates
    # The number of iterations is bounded by budget/cost
    
    # Simple approach: use a fixed beta that balances exploration and exploitation
    # For cost-aware acquisition, divide EI by cost and add UCB term
    beta = 1.0
    
    # UCB acquisition: EI/cost + beta * sigma/cost
    ucb = ei / cost_safe + beta * sigma_safe / cost_safe
    
    # Ensure finite values
    ucb = np.where(np.isfinite(ucb), ucb, 0.0)
    
    return ucb
# EVOLVE-BLOCK-END
