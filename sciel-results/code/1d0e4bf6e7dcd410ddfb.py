"""Cost-aware acquisition for rastrigin 2d BO loop."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track elapsed time and return a safe fallback if we're running out of time.
_T0 = time.time()
_DEADLINE = 325.0  # safety margin of 5s before the 330s kill


def _deadline_exceeded():
    return (time.time() - _T0) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Quantile of Improvement (QI) acquisition function.
    
    QI maximizes the conditional expected improvement given that improvement occurs,
    weighted by a probability threshold. This focuses search on regions with both
    high probability of improvement AND high expected gain, which is particularly
    effective for multimodal landscapes like rastrigin.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    if _deadline_exceeded():
        # Fallback: simple expected improvement per cost
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardize the improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement: P(f(x) < f_best)
    p_improve = norm.cdf(z)
    
    # Expected improvement
    ei = (f_best - mu) * p_improve + sigma_safe * norm.pdf(z)
    
    # Quantile of Improvement: E[f(x) | f(x) < f_best] * P(f(x) < f_best)
    # This is the expected value of the objective conditional on improvement,
    # weighted by the probability of improvement.
    # 
    # QI = E[min(f(x), f_best)] = f_best - EI
    # But we want to maximize improvement, so we use EI directly.
    # 
    # A more nuanced QI: condition on being in the top quantile of improvements.
    # We use a threshold on z to focus on high-confidence improvements.
    
    # Compute the conditional expected value given improvement
    # E[f(x) | f(x) < f_best] = f_best - EI / P(improve)
    # But this can be unstable when P(improve) is small.
    
    # Instead, use a smoothed QI that balances exploration and exploitation:
    # QI = EI * (1 + alpha * z) for z > 0 (exploitation in high-confidence regions)
    # QI = EI * (1 - beta * |z|) for z < 0 (exploration in uncertain regions)
    
    # A simpler and more robust approach: use EI with a cost adjustment
    # that accounts for the cost structure (1 + mean(x_scaled))
    
    # Cost-aware QI: divide EI by cost, but add a small exploration bonus
    # for regions with high uncertainty (high sigma) relative to the scale of f_best
    
    # The cost is 1 + mean(x_scaled to [0,1]), so cost is in [1, 2]
    # Lower cost is better, so we divide by cost
    
    # Add a small exploration term to prevent pure exploitation
    # The exploration term is proportional to sigma, normalized by the scale of the problem
    
    # For rastrigin, the minimum is 0 and values can be large.
    # Use a relative exploration term based on sigma / (|f_best| + 1)
    
    f_scale = abs(f_best) + 1.0
    exploration = sigma_safe / f_scale
    
    # Combine EI and exploration
    # Weight: 0.8 for exploitation (EI/cost), 0.2 for exploration (sigma/cost)
    # This balance helps in multimodal landscapes by not getting stuck in local optima
    
    ei_per_cost = ei / (cost + 1e-9)
    expl_per_cost = exploration / (cost + 1e-9)
    
    # Final score: weighted combination
    score = 0.8 * ei_per_cost + 0.2 * expl_per_cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
