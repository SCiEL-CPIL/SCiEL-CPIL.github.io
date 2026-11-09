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
    
    Uses a hybrid approach:
    - Early stage: UCB with higher exploration to escape local minima in multimodal landscape
    - Later stage: Blend of EI and UCB, normalized by cost
    
    The cost normalization ensures we prefer evaluations that are "cheap" in terms
    of the cost metric (1 + mean(x_scaled)).
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
    
    # Estimate progress: how many points have been evaluated so far
    # cost has shape (2000,) - these are the costs of the 2000 candidate points
    # We can use the number of finite values in mu as a proxy, but actually
    # mu and sigma are predictions for 2000 new candidates.
    # We need another way to estimate progress.
    
    # Actually, looking at the task: "While total cost is below 30, the evaluator scores
    # 2000 new uniform candidates with acquisition and evaluates the best one."
    # So each call to acquisition is for one iteration. We don't directly know the
    # total cost or iteration number.
    
    # However, we can infer something from f_best: as we make progress, f_best decreases.
    # For Griewank, the minimum is 0. If f_best is close to 0, we're in a late stage.
    
    # Alternative: use the spread of sigma. Early on, sigma is large (high uncertainty).
    # Later, sigma is smaller in explored regions.
    
    mean_sigma = np.mean(sigma)
    # Normalize sigma spread: early = high mean_sigma, late = low mean_sigma
    # Griewank values are in [0, ~1] range typically
    # A reasonable threshold: if mean_sigma > 0.1, we're in early/mid stage
    
    # Use a blend of EI and UCB
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Adaptive beta: higher exploration early, lower later
    # mean_sigma gives a sense of uncertainty
    # Scale beta based on mean_sigma: early (high sigma) -> beta ~ 1.5, late -> beta ~ 0.5
    beta = 0.5 + 1.0 * np.clip(mean_sigma / 0.1, 0, 1.0)
    
    ucb = _ucb(mu, sigma, f_best, beta)
    
    # Normalize EI and UCB to comparable scales
    # EI is in objective units, UCB is in objective units too (approximately)
    # Let's blend them
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
