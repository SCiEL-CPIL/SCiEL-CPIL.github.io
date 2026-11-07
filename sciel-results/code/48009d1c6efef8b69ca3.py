"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware LCB acquisition function for Bayesian optimization.
    Uses an adaptive exploration coefficient (beta) that decreases as f_best improves,
    normalized by evaluation cost.
    
    Score = (beta * sigma - mu) / cost
    """
    start_time = time.time()
    DEADLINE = 320.0  # Safety margin before 330s kill

    n = mu.shape[0]
    
    # Clamp sigma for numerical stability
    sigma_safe = np.clip(sigma, 1e-12, None)
    
    # Clamp cost for numerical stability
    cost_safe = np.maximum(cost, 1e-10)
    
    # Adaptive beta:
    # High f_best (early search) -> High beta (explore uncertainty)
    # Low f_best (late search) -> Low beta (exploit low mean)
    # We add 1.0 to ensure beta >= 1 even if f_best is 0 (pure exploitation of mu - sigma)
    beta = 1.0 + 1.5 * np.sqrt(max(f_best, 0.0))
    
    # Lower Confidence Bound value: mu - beta * sigma
    # We want to minimize this value.
    # Acquisition score (higher is better) is the negative of LCB divided by cost.
    # score = - (mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    scores = (beta * sigma_safe - mu) / cost_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Fallback: simple cost-aware UCB (minimize mu + k*sigma)
        # This is extremely fast, so this branch is unlikely to be triggered
        ucb = mu + 1.5 * sigma_safe
        simple = -ucb / cost_safe
        return np.where(np.isfinite(simple), simple, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
