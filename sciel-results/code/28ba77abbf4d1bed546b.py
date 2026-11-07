import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses an adaptive exploration coefficient (beta) that decreases as f_best improves,
    normalized by a weighted evaluation cost.
    
    Score = (beta * sigma - mu) / cost^0.8
    """
    start_time = time.time()
    DEADLINE = 320.0  # Safety margin before 330s kill

    # Ensure inputs are 1D arrays for consistent shape handling
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma for numerical stability
    # sigma represents standard deviation, so it should be non-negative.
    # We clamp to a small positive number to avoid division issues if used in other contexts, 
    # though here it's just multiplication.
    sigma_safe = np.clip(sigma, 1e-12, None)
    
    # Clamp cost for numerical stability
    # Cost is 1 + mean(x_scaled), so it's in [1, 2]. 
    # Clamp to a small positive number to avoid division by zero.
    cost_safe = np.maximum(cost, 1e-10)
    
    # Adaptive beta schedule:
    # - Uses log(1 + f_best) for smooth decay.
    # - When f_best is large (early), beta is large -> high exploration (sigma term dominates).
    # - When f_best is small (late, near optimum), beta -> 1.0 -> exploitation (mu term dominates).
    # - The multiplier 2.0 provides a strong initial push to explore high-uncertainty regions.
    f_best_clamped = max(float(f_best), 0.0)
    beta = 1.0 + 2.0 * np.log1p(f_best_clamped)
    
    # Lower Confidence Bound (LCB) logic for minimization:
    # We want to minimize mu - beta * sigma.
    # To maximize acquisition score (higher is better), we use:
    # score = (beta * sigma - mu)
    
    # Cost-aware weighting:
    # Dividing by cost^0.8 favors lower cost candidates more strongly than linear division.
    # This is effective when costs are in a narrow range like [1, 2].
    cost_weighted = np.power(cost_safe, 0.8)
    
    scores = (beta * sigma_safe - mu) / cost_weighted
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    # Deadline check (unlikely to be triggered given O(N) vectorized ops)
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Fallback: simple cost-aware LCB
        simple_scores = (1.5 * sigma_safe - mu) / cost_safe
        return np.where(np.isfinite(simple_scores), simple_scores, -1e9)
    
    return scores
