"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time to ensure we return before the 330s limit
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to spare before the 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Hybrid EI + UCB acquisition function with cost awareness.
    
    Combines Expected Improvement (exploitation) with an Upper Confidence
    Bound component (exploration), normalized by evaluation cost.
    The exploration weight adapts to the current optimization state:
    more exploration early (when f_best is far from optimum), less later.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value (lower is better).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard: if we're close to the limit, return a simple safe result
    elapsed = time.time() - _START_TIME
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        # Return a simple, fast, finite score as a safety fallback
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-8)
        score = np.maximum(-mu, 0.0) / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # UCB component for exploration: rewards high uncertainty
    # Adaptive beta: more exploration when f_best is large (early),
    # less when f_best is small (late, closer to optimum)
    # Levy 2D minimum is 0, so f_best approaches 0 as we converge
    f_best_val = float(f_best)
    # Scale: when f_best is large (>1), use more exploration;
    # when f_best is small (<0.1), use less exploration
    beta = 0.5 + 1.5 * np.exp(-f_best_val / 2.0)
    ucb = beta * sigma_safe - mu
    
    # Weighted combination: EI (exploitation) + UCB (exploration)
    # The UCB term helps discover new regions when EI is low
    # Adaptive weight: stronger exploration early, weaker late
    w_ucb = 0.2 + 0.4 * np.exp(-f_best_val / 2.0)
    
    # Combine: primary EI signal + secondary UCB exploration signal
    # EI is in objective units, UCB is also in objective units
    combined = ei + w_ucb * np.maximum(ucb, 0.0)
    
    # Cost-aware: divide by cost
    score = combined / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
