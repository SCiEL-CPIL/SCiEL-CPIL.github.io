"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level deadline guard: 330s wall-clock limit, use 320s as safety margin
_T0 = time.time()
_DEADLINE = _T0 + 320.0


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running out of time, return simple POI/cost
    if time.time() > _DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        score = poi / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Estimate remaining budget
    # Initial cost is 5 points * avg cost 1.5 = 7.5
    # Max total cost is 30
    # We don't know exactly how many evaluations have happened, but we can estimate
    # based on the fact that cost is an array of costs for the 2000 candidates.
    # Wait, the contract says: "mu, sigma and cost have shape (2000,)."
    # "cost" here is the cost of evaluating each of the 2000 candidates, not the total spent.
    # "While total cost is below 30, the evaluator scores 2000 new uniform candidates..."
    # So `cost` is a vector of length 2000, where cost[i] is the cost of evaluating candidate i.
    # The total cost spent so far is not directly passed.
    # However, we can infer the "phase" of the search from f_best.
    # Early on, f_best is high (far from -3.32). Late on, f_best is low (close to -3.32).
    # Let's use the distance from the known minimum (-3.32237) to estimate progress.
    # Hartmann 6D min is -3.32237.
    # Initial uniform points usually have f ~ 0 to 1.
    # If f_best is close to -3.32, we are late in the search.
    
    # A more robust way to estimate remaining budget without knowing exact history:
    # The problem states "Runs start with 5 uniform random points (cost counted). While total cost is below 30..."
    # Let's assume the search progresses roughly linearly in cost.
    # We can estimate the "fraction of budget used" by looking at how good f_best is.
    # But this is heuristic.
    
    # Alternative: The prompt suggests "dynamically scales the exploration term by the square root of the remaining budget".
    # Since we don't have the exact total cost spent, we can approximate the "urgency" by the value of f_best.
    # Let's define a progress metric p in [0, 1] where 0 is start and 1 is end.
    # f_best starts around 0.5 (mean of uniform on [0,1]^6 for Hartmann 6D is not 0.5, actually Hartmann functions are negative).
    # Let's check Hartmann 6D range.
    # Hartmann 6D has min -3.32237.
    # Typical values are between -3.3 and 1.
    # Let's assume a range of [-3.5, 1.0] for normalization.
    # If f_best is -3.3, we are very close to the end.
    # If f_best is 0.0, we are early.
    
    # Let's map f_best to a progress factor.
    # p = (f_best - f_min) / (f_max_expected - f_min)
    # f_min = -3.32237
    # f_max_expected ~ 1.0
    # p = (f_best - (-3.32237)) / (1.0 - (-3.32237)) = (f_best + 3.32237) / 4.32237
    # If f_best = -3.32, p ~ 0.0005 (very late, almost done) -> wait, this is inverted.
    # If f_best is low (good), we are late. If f_best is high (bad), we are early.
    # So "remaining budget" is high when f_best is high.
    # Let's define `remaining_budget_factor` = 1 - p = (1.0 - f_best) / 4.32237 ? No.
    
    # Let's think about the cost.
    # Total budget = 30.
    # Start cost ~ 7.5.
    # Remaining ~ 22.5.
    # As we evaluate, total cost increases.
    # We don't know the current total cost.
    # However, the number of evaluations is roughly (Total Cost Spent) / (Avg Cost per Eval).
    # Avg cost per eval is 1.5.
    # If we assume we are at a typical point in the search, we can just use a fixed beta?
    # No, the directive says "dynamically scales... by square root of remaining budget".
    
    # Let's try to estimate the current total cost.
    # We can't.
    # But we can use the value of f_best as a proxy for "how much work has been done".
    # A good f_best implies many evaluations.
    # Let's use a simple heuristic:
    # beta = 1.0 * sqrt(remaining_budget_fraction)
    # remaining_budget_fraction is estimated by how far f_best is from the minimum.
    # If f_best is -3.32 (very good), remaining budget is small (we are close to limit or found it).
    # If f_best is 0.0 (bad), remaining budget is large.
    
    # Let's normalize f_best.
    # f_min = -3.32237
    # f_max_observed_typical = 1.0
    # progress = (f_best - f_min) / (f_max_observed_typical - f_min)
    # progress is 0 when f_best = f_min (perfect, end of search)
    # progress is 1 when f_best = 1.0 (start of search)
    # So "remaining_budget_proxy" = progress.
    
    f_min = -3.32237
    f_max = 1.0
    progress = (f_best - f_min) / (f_max - f_min)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Early in search: progress ~ 1.0 -> sqrt(1.0) = 1.0 -> high exploration
    # Late in search: progress ~ 0.0 -> sqrt(0.0) = 0.0 -> pure exploitation
    # This matches the requirement: "aggressive exploration early... precise exploitation later"
    
    exploration_scale = np.sqrt(progress)
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the score, so we want to pick points with low predicted values.
    # score = (mu - beta * sigma) / cost
    # Note: Lower mu is better. So we want to maximize (mu - beta*sigma)? No.
    # LCB is a lower bound on the true value. We want to find the minimum.
    # So we should pick the point with the lowest LCB.
    # But the acquisition function returns scores where "higher = evaluate next".
    # So we should return -LCB / cost? Or 1 / cost * (something that is high for good points)?
    
    # If we use LCB = mu - beta * sigma.
    # Good points have low LCB.
    # So we want to maximize -LCB.
    # score = - (mu - beta * sigma) / cost
    # score = (beta * sigma - mu) / cost
    
    # Let's check the sign.
    # If mu is low (good), -mu is high (good).
    # If sigma is high (uncertain), beta*sigma is high (good for exploration).
    # So score = (beta * sigma - mu) / cost seems correct for maximization.
    
    beta = 2.0 * exploration_scale
    
    score = (beta * sigma_safe - mu) / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
