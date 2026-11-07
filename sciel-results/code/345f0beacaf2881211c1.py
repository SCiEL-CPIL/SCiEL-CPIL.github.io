"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d).
Strategy: Blend of Expected Improvement (EI) and Linear UCB, with adaptive exploration.
The EI component handles exploitation (improving upon f_best) while the UCB component
handles exploration (uncertainty). The blend weight adapts based on estimated progress.
Cost-awareness is applied by dividing the score by the evaluation cost.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to spare

def _is_deadline_exceeded():
    """Check if we are close to the global time limit."""
    if time.time() - _START_TIME > (300.0 - _DEADLINE_MARGIN):
        return True
    return False

def _expected_improvement(mu, sigma, f_best):
    """
    Compute Expected Improvement for minimization.
    EI(x) = E[max(f_best - f(x), 0)]
    where f(x) ~ N(mu(x), sigma(x)^2)
    """
    sigma_safe = np.clip(sigma, 1e-12, None)
    improvement = f_best - mu
    z = improvement / sigma_safe
    # EI = improvement * Phi(z) + sigma * phi(z)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI should be non-negative
    return np.clip(ei, 0.0, None)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending EI and UCB.
    
    Parameters:
    mu: array-like, mean predictions from GP
    sigma: array-like, standard deviation predictions from GP
    f_best: float, best observed value so far
    cost: array-like, cost of evaluating each candidate
    
    Returns:
    scores: array of finite scores, higher is better
    """
    # Safety check: if time is running out, return a safe random-like score
    # to allow the loop to finish or fail gracefully without hanging.
    if _is_deadline_exceeded():
        return np.zeros_like(mu, dtype=float)

    # Ensure inputs are numpy arrays for vectorization
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Numerical stability
    epsilon = 1e-8
    sigma_safe = np.clip(sigma, epsilon, None)
    cost_safe = np.clip(cost, epsilon, None)
    
    # Estimate progress based on f_best
    # Hartmann 3D minimum is -3.86278. Maximum is around 3.86 (actually lower, but let's use a safe range).
    # If f_best is close to min, we are late (exploit). If f_best is high, we are early (explore).
    f_min_expected = -3.86278
    f_max_expected = 3.86  # Rough upper bound
    
    # Normalize f_best to a progress metric [0, 1]
    # progress = 0 -> f_best = f_max (early, explore)
    # progress = 1 -> f_best = f_min (late, exploit)
    progress = (f_max_expected - f_best) / (f_max_expected - f_min_expected)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Adaptive parameters
    # Early: high exploration weight (w_ucb high), low exploitation weight (w_ei low)
    # Late: high exploitation weight (w_ei high), low exploration weight (w_ucb low)
    
    # UCB beta: controls the strength of the uncertainty bonus in the UCB term
    # Start with beta ~ 2.0, decay to ~ 0.5
    beta = 2.0 - 1.5 * progress
    
    # Blend weights: 
    # We want to mix EI and UCB.
    # EI is naturally bounded and focuses on improvement.
    # UCB (mu - beta*sigma) can be negative and focuses on lower bound.
    # A common blend is: Score = w_ei * EI + w_ucb * (f_best - mu + beta * sigma)
    # Note: (f_best - mu + beta * sigma) is the "Optimistic Improvement" or "UCB Improvement".
    # Let's use this form to keep both terms in the "improvement" scale (positive is good).
    
    # EI score
    ei_scores = _expected_improvement(mu, sigma_safe, f_best)
    
    # UCB-like improvement score: (f_best - mu + beta * sigma)
    # This is the maximum possible improvement under the UCB uncertainty bound.
    ucb_improvement = f_best - mu + beta * sigma_safe
    
    # Weights:
    # Early: w_ucb = 1.0, w_ei = 0.5
    # Late: w_ucb = 0.5, w_ei = 1.0
    w_ucb = 1.0 - 0.5 * progress
    w_ei = 0.5 + 0.5 * progress
    
    # Combined score before cost adjustment
    combined_score = w_ei * ei_scores + w_ucb * ucb_improvement
    
    # Cost-awareness: divide by cost
    # Higher cost means lower score.
    scores = combined_score / cost_safe
    
    # Final safety: replace any non-finite values with 0
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Clip to non-negative? 
    # EI is non-negative. UCB improvement can be negative if mu > f_best + beta*sigma.
    # If combined score is negative, it means we expect no improvement and high cost.
    # It's better to clip to 0 to avoid selecting points that are clearly worse than f_best
    # unless all points are worse. But if all are negative, argmax picks the least negative.
    # However, negative scores might be penalized if the evaluator expects positive scores?
    # The contract says "higher = evaluate next". Negative is lower than positive.
    # So if all are negative, the least negative (closest to 0) is picked. This is correct.
    # But to be safe and ensure we don't pick a point with *expected* worse value if there's 
    # a point with *expected* better value, we rely on the relative ordering.
    # Let's keep them as is, just ensuring finiteness.
    
    return scores
# EVOLVE-BLOCK-END
