"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of UCB and EI,
    with adaptive exploration that increases as budget depletes.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: return quickly if we're close to timeout
    # We don't have a global timer here, but we can make this fast
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB)
    # Use an adaptive beta that increases with uncertainty
    # Since we don't have the iteration count directly, we use f_best as a proxy
    # Lower f_best (better) suggests we're later in optimization if the function is being minimized
    # Actually, let's use a fixed but reasonable beta for UCB
    beta = 1.0
    ucb = mu + beta * sigma_safe
    
    # For minimization, we want low mu and high sigma
    # UCB for minimization: -mu + beta * sigma (we want to minimize, so we prefer low mu)
    # Actually, standard UCB for minimization is: -mu + beta * sigma
    # Or equivalently, we can think of it as maximizing: -(mu) + beta * sigma
    
    # Let's use a cost-aware version
    # Normalize costs to [0, 1] range
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        cost_norm = np.ones_like(cost) * 0.5
    else:
        cost_norm = (cost - cost_min) / cost_range
    
    # Cost penalty: higher cost means lower score
    # Use a soft penalty that's not too aggressive
    cost_penalty = 1.0 / (1.0 + 2.0 * cost_norm)
    
    # Hybrid: combine EI and UCB
    # EI is in objective units (positive when mu < f_best)
    # UCB for minimization: we want to maximize (-mu + beta * sigma)
    # But we need to scale them appropriately
    
    # Scale EI to be comparable to UCB
    # EI is typically in the range [0, f_best - min(mu)]
    # UCB term (-mu + beta * sigma) can be large
    
    # Let's use a weighted combination
    # Weight EI more when we're confident (low sigma), UCB more when uncertain (high sigma)
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    if sigma_max < 1e-6:
        # Very confident, use mostly EI
        w_ei = 0.9
        w_ucb = 0.1
    else:
        # Adaptive weights based on relative uncertainty
        # When sigma is high relative to improvement, favor UCB
        improvement = np.maximum(f_best - mu, 0)
        rel_uncertainty = sigma_safe / (improvement + sigma_mean + 1e-9)
        
        # Weight UCB more when relative uncertainty is high
        w_ucb = np.clip(0.3 + 0.4 * rel_uncertainty, 0.3, 0.7)
        w_ei = 1.0 - w_ucb
    
    # Compute UCB for minimization (maximize this)
    ucb_min = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    # EI is already in objective units
    # Scale UCB by the typical scale of the objective
    obj_scale = max(np.abs(f_best), 1.0)
    ucb_scaled = ucb_min / obj_scale
    
    # Combine
    scores = w_ei * ei / obj_scale + w_ucb * ucb_scaled
    
    # Apply cost penalty
    scores = scores * cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
