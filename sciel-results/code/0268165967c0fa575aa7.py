"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_ACQ_START_TIME = None
_ACQ_DEADLINE = 300.0  # Wall-clock limit


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
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate).
    """
    global _ACQ_START_TIME
    if _ACQ_START_TIME is None:
        _ACQ_START_TIME = time.time()
    
    elapsed = time.time() - _ACQ_START_TIME
    if elapsed > _ACQ_DEADLINE - 10:
        # Safety fallback: return EI/cost quickly
        sigma_safe = np.maximum(sigma, 1e-10)
        cost_safe = np.maximum(cost, 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of being a new minimum
    prob_new_min = norm.cdf(z)
    
    # UCB for minimization: we want low mu and high sigma
    # UCB score = f_best - mu + beta * sigma
    beta = 1.0
    ucb_score = f_best - mu + beta * sigma_safe
    
    # Normalize both EI and UCB by cost
    ei_cost = ei / cost_safe
    ucb_cost = ucb_score / cost_safe
    
    # Adaptive weighting based on relative uncertainty
    # When sigma is high relative to the scale of f_best, favor exploration (UCB)
    # When sigma is low, favor exploitation (EI)
    sigma_scale = np.max(sigma_safe)
    if sigma_scale > 1e-10:
        sigma_norm = sigma_safe / sigma_scale
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Weight: more UCB when sigma is high, more EI when sigma is low
    # Use a smooth transition
    w_ucb = 0.3 + 0.4 * sigma_norm
    w_ei = 1.0 - w_ucb
    
    # Combine EI and UCB
    primary = w_ei * ei_cost + w_ucb * ucb_cost
    
    # Secondary term: bonus for high probability of being new minimum
    # This encourages exploring regions where improvement is likely
    exploration_bonus = 0.1 * prob_new_min * sigma_safe / cost_safe
    
    # Tertiary term: mild penalty for very high cost to favor cheaper evaluations
    # when EI is comparable
    cost_penalty = 0.02 * (cost_safe - 1.0) ** 2
    
    # Final score
    scores = primary + exploration_bonus - cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
