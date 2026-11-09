import numpy as np
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    cost = np.asarray(cost, dtype=float).ravel()

    # Ensure numerical stability
    sigma = np.clip(sigma, 1e-12, None)
    cost = np.clip(cost, 1e-12, None)

    # Deadline guard: if we are close to the 330s limit, return simple scores quickly
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Return a simple score to avoid timeout
        score = -mu / cost
        return np.where(np.isfinite(score), score, 0.0)

    # Total budget is 30 (from task description: "While total cost is below 30")
    # Initial cost is 5 (from 5 uniform random points)
    # So remaining budget is roughly 30 - current_total_cost
    # We don't have current_total_cost directly, but we can estimate it.
    # Actually, the cost parameter is the cost of each candidate point.
    # The total cost is the sum of costs of all evaluated points.
    # We don't have the total cost passed in, only the per-point cost.
    # However, we can use a heuristic for kappa based on the number of points or just a fixed schedule.
    
    # Let's use a dynamic kappa that decreases as we assume we've spent more budget.
    # Since we don't know the exact step, let's estimate based on the number of calls? 
    # No, global state is tricky. Let's use a fixed but well-tuned kappa or a simple heuristic.
    
    # The task says "minimizes three hump camel 2d".
    # LCB: minimize mu + kappa * sigma.
    # Score: - (mu + kappa * sigma) / cost.
    
    # A common choice for kappa in BO is to scale with the number of evaluations.
    # Since we don't have n, let's try a fixed kappa first, or estimate n from time?
    # Time is a proxy.
    # Let's try kappa = 1.0 + 0.5 * (elapsed / 300) ? No, that increases exploration over time, which is wrong.
    # We want to decrease exploration over time.
    # Let's assume the process runs for a few seconds.
    # Let's try a fixed kappa that balances exploration and exploitation.
    # For minimization with cost, we want to find the global min quickly.
    
    # Let's try kappa = 1.5
    # Score = -(mu + kappa * sigma) / cost
    
    kappa = 1.5
    
    # Compute LCB
    lcb = mu + kappa * sigma
    
    # Compute cost-aware score
    # We want to maximize the score.
    # Lower LCB is better. Higher cost is worse.
    # So score = -lcb / cost
    score = -lcb / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
