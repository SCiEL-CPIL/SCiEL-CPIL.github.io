"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_ACQ_START_TIME = None
_ACQ_DEADLINE = 280.0  # Safety margin before 300s limit


def _cost_normalized_ucb(mu, sigma, f_best, cost, beta=1.0):
    """
    Cost-normalized UCB acquisition.
    Higher scores are better (we want to minimize the objective).
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # UCB for minimization: we want low mu and high sigma
    ucb = f_best - mu + beta * sigma_safe
    score = ucb / cost_safe
    
    return np.where(np.isfinite(score), score, 0.0)


def _cost_normalized_ei(mu, sigma, f_best, cost, xi=0.0):
    """
    Cost-normalized Expected Improvement.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # EI for minimization
    imp = f_best - mu - xi
    z = imp / sigma_safe
    ei = imp * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    score = ei / cost_safe
    
    return np.where(np.isfinite(score), score, 0.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI and UCB with adaptive weighting.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions for 2000 candidates.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions for 2000 candidates.
    f_best : float
        Current best observed value (lowest).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    global _ACQ_START_TIME
    if _ACQ_START_TIME is None:
        _ACQ_START_TIME = time.time()
    
    # Check deadline - if we're running out of time, use a fast fallback
    elapsed = time.time() - _ACQ_START_TIME
    if elapsed > _ACQ_DEADLINE:
        # Fast fallback: simple EI/cost
        sigma_safe = np.maximum(sigma, 1e-10)
        cost_safe = np.maximum(cost, 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Convert to numpy arrays and ensure proper types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        return np.ones(2000, dtype=np.float64)
    
    # Ensure sigma is non-negative and cost is positive
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # Compute both acquisition scores
    ei_score = _cost_normalized_ei(mu, sigma, f_best, cost, xi=0.0)
    ucb_score = _cost_normalized_ucb(mu, sigma, f_best, cost, beta=1.0)
    
    # Adaptive weighting based on relative uncertainty
    # When sigma is high relative to the scale, favor exploration (UCB)
    # When sigma is low, favor exploitation (EI)
    sigma_max = np.max(sigma)
    if sigma_max > 1e-10:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Weight: more UCB when sigma is high, more EI when sigma is low
    # Use a smooth transition with moderate exploration bias
    w_ucb = 0.3 + 0.4 * sigma_norm
    w_ei = 1.0 - w_ucb
    
    # Combine EI and UCB
    primary = w_ei * ei_score + w_ucb * ucb_score
    
    # Secondary term: bonus for high probability of being new minimum
    # This encourages exploring regions where improvement is likely
    z = (f_best - mu) / sigma
    prob_new_min = norm.cdf(z)
    exploration_bonus = 0.1 * prob_new_min * sigma / cost
    
    # Tertiary term: mild penalty for very high cost to favor cheaper evaluations
    # when EI is comparable
    cost_penalty = 0.02 * (cost - 1.0) ** 2
    
    # Final score
    scores = primary + exploration_bonus - cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
