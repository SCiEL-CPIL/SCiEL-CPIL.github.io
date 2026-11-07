"""Adaptive acquisition for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def _expected_improvement(mu, sigma, f_best, xi):
    """Compute Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-12)
    improvement = f_best - mu - xi
    z = improvement / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = improvement * cdf + sigma_safe * pdf
    # Numerical stability
    ei = np.where(np.isfinite(ei), ei, 0.0)
    return ei

def _ucb_score(mu, sigma, kappa):
    """Compute UCB score for minimization (higher is better)."""
    return -mu + kappa * sigma

def _probability_improvement(mu, sigma, f_best):
    """Compute Probability of Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-12)
    improvement = f_best - mu
    z = improvement / sigma_safe
    pi = norm.cdf(z)
    return np.where(np.isfinite(pi), pi, 0.0)

def acquisition(mu, sigma, f_best):
    """
    Adaptive hybrid acquisition function for Styblinski-Tang 1D.
    
    Uses a combination of EI and exploration bonus that adapts to the
    current state of the GP posterior. The key insight is that for
    multimodal functions like ST1D, we need strong exploration early
    and exploitation later.
    
    Strategy:
    - Compute base EI with small xi
    - Add an exploration bonus that scales with sigma and decays as
      we gain confidence (lower sigma)
    - The exploration bonus is adaptive: it's stronger when sigma is
      high relative to the improvement potential
    
    This provides a smooth transition from exploration to exploitation
    without the instability of fixed UCB parameters.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a simple greedy score to avoid timeout
        return -(mu) + 0.0
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute the spread of sigma to understand exploration state
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_min = np.min(sigma_safe)
    
    # Compute base EI with small xi for slight exploration bias
    xi = 0.01
    ei = _expected_improvement(mu, sigma_safe, f_best, xi)
    
    # Compute exploration bonus:
    # We want to explore where sigma is high, but the bonus should
    # be proportional to the potential for finding a better solution.
    # A principled approach: bonus = beta * sigma * (1 - cdf(z))
    # where z = (f_best - mu) / sigma, so (1-cdf(z)) = probability
    # that this point beats f_best.
    
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Probability that this point is better than f_best
    prob_better = 1.0 - norm.cdf(z)
    prob_better = np.where(np.isfinite(prob_better), prob_better, 0.0)
    
    # Adaptive beta: scale exploration based on how much uncertainty
    # remains relative to the best known value
    # If sigma_mean is large relative to |f_best|, we're in early exploration
    # If sigma_mean is small, we should exploit more
    
    # Normalize sigma to get a sense of exploration level
    # Use the ratio of sigma to the scale of the objective
    scale = max(abs(f_best), 1.0)
    exploration_level = np.clip(sigma_mean / scale, 0.1, 1.0)
    
    # Beta adapts: higher when exploration_level is high
    beta = 1.5 * exploration_level + 0.3
    
    # Exploration bonus: encourages visiting high-sigma regions
    # where there's still a chance of improvement
    exploration_bonus = beta * sigma_safe * prob_better
    
    # Combine: EI provides exploitation signal, bonus provides exploration
    score = ei + exploration_bonus
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
