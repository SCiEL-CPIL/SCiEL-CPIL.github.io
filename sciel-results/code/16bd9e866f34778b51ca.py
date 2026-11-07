import numpy as np
from scipy.stats import norm
import time

_deadline = None
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB (Upper Confidence Bound) acquisition function.
    
    Uses an adaptive confidence parameter that decreases as the budget is consumed,
    transitioning from exploration to exploitation. The score is:
        score = -(mu - beta * sigma) / cost
    where beta is an adaptive confidence coefficient.
    
    Parameters:
    mu: (2000,) array of mean predictions
    sigma: (2000,) array of standard deviations
    f_best: float, current best observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    (2000,) array of scores (higher is better)
    """
    global _deadline, _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0
    
    _call_count += 1
    
    if time.time() > _deadline:
        return np.ones(len(mu))
    
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Estimate how much budget has been used.
    # Total budget is 30. We start with 5 uniform points, so initial cost is roughly
    # 5 * (1 + 0.5) = 7.5. The loop runs while total cost < 30.
    # Each call to acquisition corresponds to one iteration of the loop.
    # We can estimate the fraction of budget used based on call count.
    # However, a better approach: use the magnitude of f_best to infer progress.
    # The minimum of Hartmann 6d is -3.32237. Initial f_best from 5 random points
    # is typically around -1 to -2. As we converge, f_best approaches -3.32.
    
    # Alternative: Use a time-based or call-based schedule.
    # Let's use a schedule based on the number of calls.
    # The loop runs for at most (30 - 7.5) / 1.5 ≈ 15 iterations on average.
    # So we expect ~15 calls. Let's make beta decay over ~20 calls.
    
    # Adaptive beta: start high for exploration, decay to low for exploitation.
    # beta = beta_max * exp(-decay * call_count)
    # Or use a linear decay.
    
    # Let's try: beta starts at 2.0 and decays to 0.1 over 20 calls.
    # beta = 0.1 + 1.9 * exp(-0.15 * call_count)
    
    beta = 0.1 + 1.9 * np.exp(-0.15 * _call_count)
    
    # UCB for minimization: we want to minimize, so we look for low (mu - beta*sigma)
    # The score should be higher for better candidates.
    # ucb_value = mu - beta * sigma  (lower is better)
    # score = -ucb_value / cost = (beta * sigma - mu) / cost
    
    ucb_value = mu - beta * sigma_safe
    scores = -ucb_value / (cost + eps)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
