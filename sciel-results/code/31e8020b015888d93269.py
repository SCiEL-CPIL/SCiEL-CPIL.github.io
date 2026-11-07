"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track time to enforce deadline guard
_start_time = time.time()
_DEADLINE = 320.0  # seconds, with 10s safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimizing Weierstrass 1D.
    
    Uses a hybrid approach: Upper Confidence Bound (UCB) with an exploration
    bonus that accounts for the multimodal nature of Weierstrass. The Weierstrass
    function has many local minima, so aggressive exploration is needed.
    
    We combine:
    - UCB: mu - beta * sigma (explores high-uncertainty regions)
    - A small EI-like term to bias toward promising areas
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a simple UCB score as fallback
        sigma_safe = np.maximum(sigma, 1e-12)
        return mu - 2.5 * sigma_safe
    
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # UCB for minimization: we want low mu and high sigma
    # beta controls exploration; for Weierstrass with 30 evals, aggressive exploration helps
    beta = 2.8
    
    ucb_score = -mu + beta * sigma_safe
    
    # Add a small exploitation component: prefer points where mu is low
    # This is a soft penalty for high mu values
    # Scale: mu ranges roughly around f_best, so use f_best as reference
    # We add a small bonus for points that are already promising
    exploit_bonus = -0.1 * np.maximum(mu - f_best, 0)  # penalize points worse than f_best
    
    # Combine: primary signal is UCB, secondary is mild exploitation
    score = ucb_score + exploit_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
