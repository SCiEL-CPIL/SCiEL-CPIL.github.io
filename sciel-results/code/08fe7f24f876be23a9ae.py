# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def _ensure_start():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time


def _elapsed():
    t0 = _ensure_start()
    return time.time() - t0


def _remaining():
    return 320.0 - _elapsed()  # safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function for Bayesian optimization.
    
    Parameters:
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - best observed value so far
    cost: array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    scores: array of shape (2000,) - finite scores, higher = evaluate next
    """
    # Deadline guard: if we're running out of time, return a simple score
    if _remaining() < 5.0:
        # Fallback: just use negative cost as a simple score
        return -np.asarray(cost, dtype=np.float64)

    # Ensure inputs are float arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: sample a single scalar theta from N(0, 1)
    # For each candidate, the sampled value is mu[i] + sigma[i] * theta
    # We want to minimize, so we score by -sampled_value / cost
    # This naturally balances exploration (high sigma -> high variance in theta)
    # and exploitation (low mu -> lower sampled value)
    
    # Sample theta from standard normal
    theta = np.random.standard_normal()
    
    # Compute sampled objective values
    sampled_values = mu + sigma_safe * theta
    
    # Cost-normalized score: we want higher score for better (lower) objective per cost
    # So score = -sampled_value / cost
    cost_safe = np.maximum(cost, 1e-6)
    scores = -sampled_values / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
