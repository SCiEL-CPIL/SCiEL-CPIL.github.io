"""Acquisition function for Bayesian optimization of styblinski tang 1d.
Uses Lower Confidence Bound (LCB) for minimization, which is equivalent
to maximizing the Upper Confidence Bound (UCB) score: -mu + beta * sigma.
This provides a linear balance between exploitation (low mu) and exploration
(high sigma), often more robust for multimodal functions than Expected
Improvement which can be overly greedy.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 300.0  # 300s wall-clock limit per evaluation

# Exploration parameter for UCB.
# EI behaves like UCB with beta ~ 0.4 in the exploration regime.
# We choose beta = 1.5 to encourage stronger exploration, helping to
# escape local minima in the Styblinski-Tang function.
_BETA = 1.5


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points using UCB.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidates (in objective units)
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidates (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Deadline safety check
    if time.time() > _DEADLINE - 5.0:
        # Fallback: Simple UCB-like score to ensure return before timeout
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + _BETA * sigma_safe

    # Ensure sigma is strictly positive to avoid numerical issues, though
    # for UCB it's just multiplication, so 0 is fine. But we clip for safety
    # if we were doing division (we aren't, but good practice).
    sigma_safe = np.maximum(sigma, 0.0)
    
    # UCB score for minimization:
    # We want to minimize f(x).
    # The lower bound on f(x) is mu(x) - beta * sigma(x).
    # To maximize the chance of finding a low value, we maximize the "optimistic"
    # potential, which is equivalent to minimizing the lower bound?
    # Actually, standard UCB for minimization:
    # Select x that minimizes mu(x) - beta * sigma(x).
    # The acquisition function (to be maximized) is therefore:
    # score(x) = -(mu(x) - beta * sigma(x)) = -mu(x) + beta * sigma(x)
    
    scores = -mu + _BETA * sigma_safe
    
    # Numerical cleanup: ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
