"""Cost-aware acquisition for Bayesian optimization on powell 4d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level state to track time budget
_start_time = time.time()
_TIME_LIMIT = 295.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled_to_01))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values (higher = more desirable)
    """
    # Check time budget
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Return a simple fallback if we're running out of time
        return np.zeros_like(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Normalize cost to [0, 1]
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB (Lower Confidence Bound for minimization)
    # We want to explore regions with high uncertainty
    # UCB score = -mu + beta * sigma (higher is better for exploration)
    # But we need to balance with exploitation
    
    # Adaptive beta based on progress
    # Early on: more exploration (higher beta)
    # Later: more exploitation (lower beta)
    # f_best starts high for Powell 4d and decreases toward 0
    progress = np.clip(1.0 - f_best / 20.0, 0.0, 1.0)
    
    # Beta decreases from ~2.0 to ~0.5 as progress increases
    beta = 2.0 - 1.5 * progress
    
    # UCB score: lower mu (better) + uncertainty bonus
    ucb = -mu + beta * sigma
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    if ucb_max > ucb_min:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Blend EI and UCB: early on more UCB (exploration), later more EI (exploitation)
    # Weight for UCB decreases with progress
    ucb_weight = 0.7 - 0.4 * progress  # From 0.7 to 0.3
    ei_weight = 1.0 - ucb_weight       # From 0.3 to 0.7
    
    # Combined acquisition before cost penalty
    combined = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Cost-aware penalty: penalize expensive evaluations
    # The penalty should be stronger when we have budget to spare
    # and weaker when we're close to the end
    # Use a soft penalty that scales with cost
    cost_penalty = 0.3 * cost_norm
    
    # Final score: combined acquisition minus cost penalty
    scores = combined - cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
