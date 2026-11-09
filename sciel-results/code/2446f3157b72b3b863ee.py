"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: track elapsed time to ensure we return well before the 330s limit
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin


def _check_deadline():
    """Return True if we should abort and return a safe fallback."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a hybrid EI + UCB approach:
    - EI for exploitation (points likely to improve over f_best)
    - UCB for exploration (points with high uncertainty)
    - Small sigma bonus to encourage exploration in early stages
    
    For minimization: higher score means more likely to be evaluated next.
    """
    # Deadline check: if we're too close to the limit, return a simple score
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei + 0.1 * sigma_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound for minimization
    # We want to explore where mu - beta*sigma is low (potentially good)
    # Score contribution: -(mu - beta*sigma) = -mu + beta*sigma
    beta = 1.2
    ucb_score = -mu + beta * sigma_safe
    
    # Weighted combination: balance exploitation and exploration
    w_ei = 0.5
    w_ucb = 0.5
    
    score = w_ei * ei + w_ucb * ucb_score
    
    # Small bonus for uncertainty to help escape local optima in early stages
    # This is a mild exploration incentive
    score += 0.05 * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
