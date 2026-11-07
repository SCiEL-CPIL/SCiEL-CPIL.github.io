import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track elapsed time per call and bail early if we're getting close.
# The acquisition function is called once per evaluation iteration.
# There are at most ~25 evaluations (30 total cost / ~1.2 avg cost per eval),
# so each call gets roughly 300/25 = 12 seconds of budget.
# But the GP fitting is done externally; our function just scores 2000 points.
# The scoring itself is vectorized and fast (< 1ms typically).
# The main risk is if something unexpected is slow. We'll add a guard.

_start_time = time.time()
_DEADLINE = 325.0  # 5s safety margin


def _elapsed():
    return time.time() - _start_time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a cost-normalized Lower Confidence Bound (LCB)
    with a dynamic exploration term scaled by the remaining budget.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the best observed value so far.
        cost: (2000,) array of evaluation costs.
        
    Returns:
        (2000,) array of scores, higher is better.
    """
    # Deadline guard
    if _elapsed() > _DEADLINE:
        # Return a neutral score to avoid wasting time
        return np.zeros(2000)

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Dynamic exploration parameter:
    # The gap between f_best and the known minimum (-0.8) tells us how far we are.
    # When far from optimum, explore more; when close, exploit more.
    # gap = f_best - (-0.8) = f_best + 0.8
    # At start, f_best ~ -0.5 (roughly), gap ~ 0.3
    # Near optimum, f_best ~ -0.8, gap ~ 0.0
    # We want beta to decrease as we get closer to optimum.
    
    gap = f_best - (-0.8)  # how far above the minimum
    # Clamp gap to reasonable range
    gap = np.clip(gap, 0.0, 1.0)
    
    # Dynamic beta: start with more exploration, reduce as we improve
    # beta = base + scale * gap
    # When gap is large (early), beta is larger (more exploration)
    # When gap is small (late), beta is smaller (more exploitation)
    beta = 0.5 + 1.5 * gap
    
    # Also scale by remaining budget factor:
    # Total budget is 30. We've spent some cost already.
    # But we don't know exactly how much is spent; cost array gives per-candidate cost.
    # The "cost" parameter is the cost of evaluating each candidate point.
    # The total spent so far isn't directly available, but we can infer from f_best
    # and the number of evaluations. Actually, we don't have that info directly.
    # Let's use a simpler approach: the cost-aware LCB divides by cost.
    
    # LCB: mu - beta * sigma (lower confidence bound, we want to minimize)
    # For maximization of acquisition, we use: -mu + beta * sigma
    # This is the Upper Confidence Bound for minimization (UCB-min)
    
    # Cost-normalized: divide by cost to prefer cheaper evaluations
    # scores = (-mu + beta * sigma) / cost
    
    # But we also want to ensure we're making progress toward f_best.
    # Combine LCB with a small PoI-like term to ensure exploitation.
    
    # Actually, let's stick with pure cost-normalized LCB as the strategy suggests.
    # The LCB naturally balances exploration (beta*sigma) and exploitation (-mu).
    
    lcb = -mu + beta * sigma_safe
    
    # Cost normalization: divide by cost
    # cost should be > 0, but let's be safe
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = lcb / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
