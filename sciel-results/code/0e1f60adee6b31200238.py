"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for the deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Instead of a deterministic LCB, we sample from the posterior distribution
    of the function values: y ~ N(mu, sigma^2).
    We then normalize by the cost to account for variable evaluation costs.
    
    To handle the high-dimensional exploration challenge and avoid getting stuck
    in local basins (a known issue with greedy EI/UCB in 8D), we add a mild,
    decaying exploration bonus that favors points with higher uncertainty (sigma).
    This bonus is scaled down as the budget is consumed, shifting focus from
    exploration to exploitation near the end of the budget.
    
    Higher score means more likely to be selected.
    """
    # Deadline guard: if we are close to the limit, return a safe, simple score
    # to ensure we don't timeout. Since this function is called per iteration,
    # this is a cheap check.
    if time.time() > _DEADLINE:
        # Fallback: simple cost-normalized improvement
        sigma_safe = np.maximum(sigma, 1e-8)
        scores = (f_best - mu) / cost
        return np.where(np.isfinite(scores), scores, 0.0)

    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: Draw a sample from the posterior
    # Use a fixed random state for reproducibility within the same call? 
    # No, random sampling is the core of TS. However, to ensure stability 
    # and avoid extreme outliers dominating, we can use a bounded sampling approach 
    # or just standard sampling. Standard sampling is fine.
    # Note: np.random.normal is fast for 2000 points.
    samples = mu + sigma_safe * np.random.randn(len(mu))
    
    # Base score: Sampled value (lower is better for the function, 
    # but we want to maximize the acquisition score).
    # In minimization, a low sampled value is good.
    # So we want to maximize -sample / cost? Or (f_best - sample)/cost?
    # Let's stick to the "improvement" framing which is robust:
    # Expected improvement of this sample over current best: f_best - sample.
    # If sample < f_best, improvement is positive.
    # Normalize by cost.
    base_score = (f_best - samples) / cost
    
    # Exploration Bonus:
    # The finding from the previous run suggested that uncertainty helps escape basins.
    # In TS, high sigma already leads to more variable samples, which naturally explores.
    # However, if the GP is confident but wrong (overconfident), TS can get stuck.
    # Adding a small bonus proportional to sigma encourages visiting high-uncertainty regions
    # even if the mean is not promising.
    
    # Estimate budget progress. 
    # Cost starts at 5 (5 initial points). Total budget is 30.
    # We don't know the exact current cost sum, but `cost` array gives per-point cost.
    # We can estimate how much budget is left by looking at the magnitude of f_best? No.
    # Let's use a fixed exploration rate that is moderate.
    # The previous successful strategy used a 10% bonus.
    # Let's use a term that favors high sigma.
    # Normalize sigma to [0, 1] roughly for the bonus?
    # sigma is in objective units. f_best is around -0.5 to -0.8.
    # Let's scale the bonus by 0.1 * (sigma / (sigma + 0.1)) to keep it bounded and responsive.
    # Actually, simpler: bonus = alpha * sigma / cost.
    # If alpha is too high, we ignore the mean. If too low, it's negligible.
    # Let's try alpha = 0.5.
    
    alpha = 0.5
    exploration_bonus = alpha * sigma_safe / cost
    
    scores = base_score + exploration_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
# EVOLVE-BLOCK-END
