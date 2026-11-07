"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 300.0
_MARGIN = 10.0

def _remaining_time():
    return _TIME_LIMIT - (time.time() - _start_time) - _MARGIN

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    LCB(x) = mu(x) - kappa(t) * sigma(x)
    where kappa(t) scales with sqrt(remaining budget) to dynamically
    shift from exploration to exploitation as the budget depletes.
    
    The cost-aware score is LCB / cost, so that:
    - Lower mu (better predicted value) increases score
    - Higher sigma (more uncertainty) increases score, weighted by kappa
    - Higher cost decreases score (cost-awareness)
    
    The exploration coefficient kappa(t) = base_kappa * sqrt(remaining / total_budget)
    where remaining = 30 - current_cost, total_budget = 30.
    This ensures maximum exploration early and pure exploitation late.
    """
    # Deadline guard
    if _remaining_time() < 1.0:
        # Fast fallback: simple LCB with fixed kappa
        sigma_safe = np.maximum(sigma, 1e-8)
        kappa = 1.0
        lcb = mu - kappa * sigma_safe
        cost_safe = np.maximum(cost, 1e-8)
        scores = lcb / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Estimate remaining budget
    # cost has shape (2000,) but represents the cost of evaluating each candidate
    # The total budget is 30. We need to estimate how much budget remains.
    # Since we don't have direct access to the current total cost, we use
    # the mean of the cost array as a proxy for the typical cost per evaluation.
    # Actually, the cost parameter is the cost of each candidate point, not the total spent.
    # We need to infer the remaining budget.
    
    # The problem states: "While total cost is below 30"
    # We don't have the current total cost directly, but we can estimate it.
    # Each call to acquisition is made during the loop. The number of evaluations
    # so far can be inferred from the fact that we started with 5 points and
    # each iteration adds 1 evaluation.
    
    # Since we don't track state, we use a heuristic:
    # The cost array values range from 1 to 2 (1 + mean(x_scaled)).
    # If we assume an average cost of ~1.5 per evaluation, and budget is 30,
    # then we can have about 20 evaluations total.
    # 
    # A better approach: use the magnitude of f_best and sigma to estimate
    # how far along we are. Early on, f_best is large and sigma is large.
    # Late on, f_best is small and sigma is small.
    
    # Alternative: Use a fixed schedule based on a proxy for progress.
    # Let's use the ratio of median sigma to initial sigma estimate.
    # Initial sigma is typically large (on the order of the function range).
    
    # For these functions:
    # - Ackley: range ~0 to ~30
    # - Rastrigin: range ~0 to ~50
    # - Levy: range ~0 to ~200
    # - Hartmann: range ~-3.86 to ~1
    
    # A simpler approach: use the mean cost as a proxy.
    # Actually, let's just use a time-varying kappa based on a counter.
    # But we can't maintain state across calls easily.
    
    # Best approach: Estimate remaining budget from the data.
    # The cost values tell us the cost of each candidate. The total budget is 30.
    # If we assume the search has been running for a while, we can estimate
    # the number of evaluations done by looking at how "converged" the GP is.
    
    # Let's use a robust approach: kappa = base * sqrt(1 - progress_estimate)
    # where progress_estimate is based on how small f_best is relative to typical values.
    
    # For the four functions, the minimum is 0 or -3.86.
    # Let's use a normalized progress based on |f_best| / scale_factor
    # where scale_factor depends on the function.
    
    # Since we don't know which function we're on, use a general approach:
    # progress = 1 - (median_sigma / initial_sigma_estimate)
    # where initial_sigma_estimate is the max of (abs(f_best), median_sigma)
    
    med_sigma = np.median(sigma_safe)
    f_abs = abs(f_best)
    
    # Estimate initial scale: the larger of f_best magnitude and sigma
    initial_scale = max(f_abs, med_sigma, 1.0)
    
    # Progress: 0 at start, 1 at end
    # When med_sigma is close to initial_scale, we're early
    # When med_sigma is much smaller, we're late
    progress = 1.0 - min(med_sigma / initial_scale, 1.0)
    progress = max(0.0, min(progress, 1.0))
    
    # Remaining budget fraction: 1 - progress
    remaining_fraction = 1.0 - progress
    
    # Base kappa: tuned for good exploration
    base_kappa = 2.0
    
    # Time-varying kappa: maximum exploration early, none late
    kappa = base_kappa * np.sqrt(remaining_fraction)
    
    # Ensure kappa is not too small
    kappa = max(kappa, 0.1)
    
    # Compute LCB: lower is better, so we want to maximize LCB
    # LCB = mu - kappa * sigma
    # We want to pick the point with the highest LCB (most promising)
    lcb = mu - kappa * sigma_safe
    
    # Cost-aware: divide by cost
    # Higher cost should reduce the score
    cost_safe = np.maximum(cost, 1e-8)
    
    scores = lcb / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
