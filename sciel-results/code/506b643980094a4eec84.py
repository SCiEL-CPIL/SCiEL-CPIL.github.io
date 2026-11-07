"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions for 2000 candidate points.
    sigma : array of shape (2000,)
        GP standard deviation predictions for 2000 candidate points.
    f_best : float
        Best observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = better candidate to evaluate).
    """
    # Deadline guard
    t_start = time.time()
    
    # Handle edge cases: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute LCB (higher is better for minimization, so we negate)
    # LCB = mu - beta * sigma, we want to maximize this, so we use -LCB = -mu + beta*sigma
    # But for acquisition, higher score = better, so we want to maximize improvement
    # LCB for minimization: mu - beta*sigma, we want to minimize this
    # So acquisition score for LCB = -(mu - beta*sigma) = beta*sigma - mu
    beta = 2.0
    lcb_score = beta * sigma_safe - mu
    
    # Hybrid: combine EI and LCB
    # Normalize both to have similar scale
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    ei_norm = (ei - ei_mean) / ei_std
    
    lcb_mean = np.mean(lcb_score)
    lcb_std = np.std(lcb_score) + 1e-9
    lcb_norm = (lcb_score - lcb_mean) / lcb_std
    
    # Combine with adaptive weighting
    # Early on, uncertainty is high, so LCB should dominate
    # Later, EI should dominate
    # Use a simple heuristic: weight based on the ratio of sigma to (f_best - mu)
    # When uncertainty is high relative to potential improvement, favor LCB
    
    # Compute a measure of exploration vs exploitation
    # If f_best is much larger than mu, exploitation is important
    # If sigma is large, exploration is important
    
    # Simple adaptive weight: 
    # w = 0.5 + 0.5 * (mean_sigma / (mean_f_best - mean_mu + eps))
    # This gives more weight to LCB when uncertainty is high
    
    mean_sigma = np.mean(sigma_safe)
    mean_improvement = f_best - np.mean(mu)
    eps = 1e-9
    
    # Weight between 0.3 and 0.7
    ratio = mean_sigma / (abs(mean_improvement) + eps)
    w_lcb = 0.3 + 0.4 * min(ratio / (1 + ratio), 1.0)
    w_ei = 1.0 - w_lcb
    
    # Combine normalized scores
    combined = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Cost-aware: divide by cost
    scores = combined / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Add a small bonus for points with low cost to encourage exploration of cheap points
    # This helps in early iterations when we want to sample cheap regions
    cost_mean = np.mean(cost_safe)
    cost_std = np.std(cost_safe) + 1e-9
    cost_norm = (cost_safe - cost_mean) / cost_std
    
    # Small cost bonus (negative for high cost, positive for low cost)
    cost_bonus = -0.1 * cost_norm
    
    scores = scores + cost_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
