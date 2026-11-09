"""Rewritten program with a structural change to a UCB-EI hybrid with adaptive exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill
_n_calls = 0
_TOTAL_EXPECTED_CALLS = 25  # 30 total evaluations - 5 initial points

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Returns 2000 scores, higher meaning evaluate next.
    Uses a hybrid of Expected Improvement (EI) and an exploration bonus (proportional to sigma).
    The exploration bonus (kappa) decays over the course of the optimization.
    """
    global _n_calls, _start_time
    
    # Deadline guard: if we are running out of time, fall back to a fast, simple EI
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        eps = 1e-9
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0)
    
    _n_calls += 1
    
    # Calculate progress (0 to 1)
    # Cap progress at 1.0 to handle edge cases
    progress = min(_n_calls / _TOTAL_EXPECTED_CALLS, 1.0)
    
    # Adaptive kappa for exploration
    # Start with high exploration (kappa ~ 1.5), end with low exploration (kappa ~ 0.2)
    # Using a quadratic decay for a smoother transition
    kappa = 0.2 + 1.3 * (1.0 - progress) ** 2
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Expected Improvement (EI) for minimization with xi=0
    # This provides the exploitation signal
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute exploration bonus
    # Higher sigma means more uncertainty, so we add kappa * sigma to the score
    exploration_bonus = kappa * sigma_safe
    
    # Total score: EI + Exploration Bonus
    # This effectively creates a UCB-like behavior but with the probabilistic weighting of EI
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
