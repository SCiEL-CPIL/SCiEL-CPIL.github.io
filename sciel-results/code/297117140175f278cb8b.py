"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    Uses a structural change: global budget-aware Expected Improvement
    where the exploration-exploitation balance shifts dynamically based
    on remaining budget ratio, rather than per-candidate cost normalization.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization: E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Global budget-aware penalty:
    # cost is the per-evaluation cost (1 + mean(x scaled to [0,1]), so in [1, 2])
    # We use the mean cost as a proxy for how much budget is being consumed per step.
    # The total budget is 30, starting cost is ~5 (5 uniform points with avg cost ~1.5).
    # We estimate remaining budget fraction to modulate exploration.
    
    # Estimate remaining budget: total budget 30, initial 5 points cost ~7.5 (5 * 1.5)
    # We don't know exactly how many evaluations have been done, but we can use
    # the mean cost as an indicator. Lower mean cost = we've been choosing cheap points.
    mean_cost = np.mean(cost)
    
    # Use a global scaling factor based on the average cost level.
    # Early on, mean_cost might be higher (random points), later it decreases.
    # We want to shift from exploration (high sigma weight) to exploitation.
    
    # Structural change: instead of dividing by per-candidate cost^alpha,
    # we subtract a term that grows as we consume budget.
    # The key insight: as total cost consumed increases, we should be more
    # aggressive about exploitation (lower beta for exploration).
    
    # Estimate total cost consumed: we can't know exactly, but we can use
    # the relationship between f_best and the initial random points.
    # A simpler approach: use the mean cost as a proxy for budget depletion.
    # If mean_cost is close to 1, we've been picking very cheap points (exploiting).
    # If mean_cost is close to 2, we've been picking expensive points.
    
    # Dynamic beta: starts high (exploration) and decreases as budget depletes.
    # We estimate budget depletion from the mean cost trend.
    # Budget fraction remaining: roughly (30 - consumed) / 30
    # consumed ≈ n_evals * mean_cost, but we don't know n_evals.
    
    # Alternative: use the spread of costs as a signal.
    # If costs are all similar, we're in a consistent phase.
    cost_std = np.std(cost)
    
    # Use a sigmoid-based transition from exploration to exploitation.
    # The "temperature" decreases as the average cost decreases (we're being smarter).
    # This is a proxy for how much we've learned.
    
    # Beta controls the weight of sigma in the score.
    # score = ei - beta * sigma * cost_factor
    # Higher beta = more exploration (penalize high uncertainty more).
    
    # Dynamic beta: high early, low late.
    # Use the mean cost as a proxy: lower mean cost = later stage = lower beta.
    # Map mean_cost from [1, 2] to beta from [1.0, 0.1]
    beta = 1.0 - 0.9 * (mean_cost - 1.0)  # beta in [0.1, 1.0]
    beta = np.clip(beta, 0.1, 1.0)
    
    # Also scale by the budget: if we've consumed a lot, be more exploitative.
    # Use the ratio of current f_best improvement to initial scale.
    # Powell 4D minimum is 0, typical random values are much larger.
    # If f_best is small, we're close to the minimum -> exploit more.
    f_best_scale = max(abs(f_best), 1e-6)
    # If f_best is small (close to 0), reduce beta further.
    beta *= np.clip(1.0 / (1.0 + f_best_scale), 0.3, 1.0)
    
    # Compute the acquisition score: EI minus exploration penalty
    # The penalty is proportional to sigma * cost, weighted by beta.
    # This makes expensive, uncertain points less attractive.
    score = ei - beta * sigma_safe * cost
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
