"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: evaluator kills after 330s. We track elapsed time for safety.
_START_TIME = time.time()
DEADLINE = _START_TIME + 315.0  # 15s safety margin


def _expected_improvement(mu, sigma, f_best):
    """Compute Expected Improvement (EI) for minimization."""
    sigma_safe = np.maximum(sigma, 1e-9)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI is 0 when improvement <= 0 and sigma is tiny
    ei = np.maximum(ei, 0.0)
    return ei


def _ucb(mu, sigma, f_best, beta):
    """Compute UCB for minimization (lower is better, so we maximize -mu + beta*sigma)."""
    return -mu + beta * sigma


def _pi(mu, sigma, f_best):
    """Probability of Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Griewank 2D.
    
    Uses a structural change: a hybrid that blends EI and UCB, with the blend
    weight driven by uncertainty (mean sigma) rather than a fixed schedule.
    This is a different design family from pure PI/cost or pure EI/cost.
    """
    # Check deadline - if we're close to timeout, return a safe fallback
    if time.time() > DEADLINE:
        # Fallback: just use PI / cost as a simple safe option
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        scores = pi / np.maximum(cost, 1e-9)
        return scores

    n = len(mu)
    
    # Estimate progress via uncertainty: early = high mean_sigma, late = low mean_sigma
    mean_sigma = np.mean(sigma)
    
    # Compute EI and UCB
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Adaptive beta: higher exploration early, lower later
    beta = 0.5 + 1.0 * np.clip(mean_sigma / 0.1, 0, 1.0)
    
    ucb = _ucb(mu, sigma, f_best, beta)
    
    # Weight: early -> more UCB, late -> more EI
    w_ucb = np.clip(mean_sigma / 0.1, 0.2, 0.8)
    w_ei = 1.0 - w_ucb
    
    combined = w_ei * ei + w_ucb * ucb
    
    # Normalize by cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = combined / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
