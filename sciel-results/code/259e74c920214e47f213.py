"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware LCB acquisition function for Bayesian optimization.
    
    Uses a dynamic exploration coefficient that adapts to the current best value,
    and normalizes by evaluation cost to prioritize high-value, low-cost candidates.
    
    Score = (beta * sigma - mu) / cost^alpha
    where beta adapts based on f_best and alpha controls cost sensitivity.
    """
    start_time = time.time()
    DEADLINE = 315.0  # Safety margin before 330s kill

    # Ensure inputs are 1D arrays for consistent shape handling
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Clamp sigma for numerical stability
    sigma_safe = np.clip(sigma, 1e-12, None)

    # Clamp cost for numerical stability
    # Cost is 1 + mean(x_scaled), so it's in [1, 2].
    cost_safe = np.maximum(cost, 1e-10)

    # Adaptive beta schedule:
    # - When f_best is large (early), beta is larger -> more exploration
    # - When f_best is small (late, near optimum), beta decreases -> more exploitation
    # Using a smooth decay based on f_best
    f_best_clamped = max(float(f_best), 0.0)
    
    # Beta starts around 2.5 and decays to ~1.0 as f_best approaches 0
    # log1p(f_best) is smooth and well-behaved
    beta = 1.0 + 1.5 * np.log1p(f_best_clamped)
    
    # Cost-aware weighting:
    # Dividing by cost^alpha favors lower cost candidates
    # alpha=0.7 provides moderate cost sensitivity
    cost_weighted = np.power(cost_safe, 0.7)

    # Lower Confidence Bound (LCB) logic for minimization:
    # We want to minimize mu - beta * sigma
    # To maximize acquisition score (higher is better), we use:
    # score = (beta * sigma - mu) / cost^alpha
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
# EVOLVE-BLOCK-END
