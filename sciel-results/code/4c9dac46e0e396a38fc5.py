"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline tracking
_acquisition_start_time = None
_DEADLINE_SECONDS = 290.0  # Safety margin before 300s limit

def _ensure_deadline():
    """Check if we're approaching the deadline."""
    global _acquisition_start_time
    if _acquisition_start_time is None:
        return True
    elapsed = time.time() - _acquisition_start_time
    return elapsed < _DEADLINE_SECONDS

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a structural change:
    Combines Expected Improvement with an uncertainty-weighted exploration bonus,
    normalized by a power of cost to balance exploitation and exploration.
    
    Parameters:
    mu: array of shape (N,) - GP mean predictions
    sigma: array of shape (N,) - GP standard deviation predictions
    f_best: float - Best observed value so far
    cost: array of shape (N,) - Cost of evaluating each point
    
    Returns:
    scores: array of shape (N,) - Acquisition scores (higher is better)
    """
    global _acquisition_start_time
    _acquisition_start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is standard normal CDF and phi is standard normal PDF
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Uncertainty bonus: rewards high uncertainty regions
    # This encourages exploration in areas with high predictive variance
    uncertainty_bonus = sigma_safe
    
    # Cost normalization: use a power less than 1 to be less aggressive
    # about penalizing expensive points, allowing some exploration
    cost_safe = np.maximum(cost, 1e-9)
    cost_penalty = np.power(cost_safe, 0.5)  # Square root of cost
    
    # Combine EI and uncertainty with a weight
    # The weight balances exploitation (EI) vs exploration (uncertainty)
    # Use a small weight for uncertainty to primarily follow EI but allow exploration
    exploration_weight = 0.1
    combined = ei + exploration_weight * uncertainty_bonus
    
    # Cost-aware score: combined improvement divided by cost penalty
    scores = combined / cost_penalty
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline check: if we're close to the limit, return early with current scores
    if not _ensure_deadline():
        # Return a simplified but valid result
        return scores
    
    return scores
# EVOLVE-BLOCK-END
