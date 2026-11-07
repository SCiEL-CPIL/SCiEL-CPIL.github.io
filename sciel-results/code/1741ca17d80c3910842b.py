"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global deadline guard: keep a safety margin before the 300 s kill.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 20 s margin


def _time_guard():
    """Return True if we are within the safe time budget."""
    return time.time() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    Returns 2000 finite scores; higher = evaluate next.
    """
    # Deadline guard: if running out of time, return a simple valid fallback.
    if not _time_guard():
        # Fallback: favor cheap, uncertain points
        c = np.maximum(cost, 1e-9)
        s = np.maximum(sigma, 1e-9)
        return s / c

    eps = 1e-9
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)

    # Thompson Sampling: Sample from the posterior distribution
    # y ~ N(mu, sigma^2)
    # Using standard normal random numbers for efficiency
    randn = np.random.randn(len(mu))
    y = mu + s * randn

    # Calculate improvement over f_best (minimization)
    # If y < f_best, we have an improvement of f_best - y
    improvement = np.maximum(f_best - y, 0.0)
    
    # Cost-aware score: Improvement per unit cost
    score = improvement / c

    # Add an exploration term to ensure diverse search when improvements are zero
    # This prevents the algorithm from getting stuck if all samples are worse than f_best
    # The exploration term is proportional to uncertainty (sigma) and inversely proportional to cost
    kappa = 0.5
    score_explore = kappa * s / c

    # Combine
    score = score + score_explore

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
