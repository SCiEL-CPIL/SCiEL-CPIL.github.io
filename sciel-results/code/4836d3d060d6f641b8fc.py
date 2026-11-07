import numpy as np
import time

# Global state for deadline guard and progress estimation
_start_time = None
_call_count = 0
_total_cost_accumulated = 0.0

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _call_count, _total_cost_accumulated
    
    # Initialize time on first call
    if _start_time is None:
        _start_time = time.time()
    
    _call_count += 1
    
    # Deadline guard: if we are close to the 330s limit, return safe exploitation scores
    # to avoid any potential overhead or instability near the cutoff.
    elapsed = time.time() - _start_time
    if elapsed > 315:
        # Simple exploitation: favor low mu, penalized by cost
        scores = (f_best - mu) / np.maximum(cost, 1e-9)
        # Clip to ensure finite
        scores = np.clip(scores, -1e6, 1e6)
        return scores
    
    # Estimate progress based on cost consumed
    # Total budget is 30. We can estimate how much cost has been used.
    # The cost array contains the cost of each candidate point.
    # We don't know exactly how many evaluations have been done, but we can
    # use the number of calls as a proxy and assume average cost.
    # Better: track cumulative cost from previous calls.
    # Since we don't have direct access to past costs, use call count.
    # Budget: 30 total. Initial 5 points cost ~5 * 1.5 = 7.5 avg.
    # Remaining budget ~22.5. Avg cost per eval ~1.5. So ~15 more evals.
    # Total calls ~20.
    max_calls = 25
    progress = min(1.0, _call_count / max_calls)
    
    # Use a smoother beta schedule: start high for exploration, decay to exploitation
    # Beta controls exploration-exploitation tradeoff
    beta_max = 3.0
    beta_min = 0.8
    # Use quadratic decay for faster initial exploration
    progress_sq = progress * progress
    beta = beta_max - (beta_max - beta_min) * progress_sq
    
    # Cost-aware UCB for minimization
    # Expected improvement-like: (f_best - mu) + beta * sigma
    # Normalize by cost to prefer cheaper points
    base_score = (f_best - mu + beta * sigma)
    
    # Add a small bonus for points with lower uncertainty to avoid noise
    # This helps when sigma is very small (converged regions)
    sigma_bonus = 0.1 * sigma / (np.mean(sigma) + 1e-9)
    
    scores = (base_score + sigma_bonus) / np.maximum(cost, 1e-9)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
