"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining EI and PI with adaptive weighting.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost to evaluate each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 280.0  # Safety margin before 300s limit
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if mu.size == 0:
        return np.array([])
    
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Compute Probability of Improvement (PI)
    # PI = Phi(z)
    pi = norm.cdf(z_clipped)
    
    # Cost normalization: use median cost as reference
    median_cost = np.median(cost)
    cost_norm = cost / (median_cost + 1e-10)
    
    # Adaptive weighting based on how much improvement is expected
    # When EI is large relative to sigma, we're in exploitation mode
    # When EI is small, we're in exploration mode
    # Use a smooth transition based on the ratio of EI to sigma
    ratio = np.abs(ei) / (sigma_safe + 1e-10)
    # Weight for EI: higher when ratio is large (exploitation)
    w_ei = 1.0 / (1.0 + np.exp(-ratio + 1.0))
    # Weight for PI: higher when ratio is small (exploration)
    w_pi = 1.0 - w_ei
    
    # Combine EI and PI with adaptive weights
    # Scale EI by a factor to make it comparable to PI
    # EI can be much larger than PI, so we normalize
    ei_scaled = ei / (np.max(ei) + 1e-10) if np.max(ei) > 1e-10 else ei
    
    combined = w_ei * ei_scaled + w_pi * pi
    
    # Divide by cost to get cost-aware score
    scores = combined / cost_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
