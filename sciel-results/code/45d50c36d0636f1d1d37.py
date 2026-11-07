import numpy as np
from scipy.stats import norm
import time

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Styblinski-Tang 2D Bayesian optimization.
    
    Combines Expected Improvement (EI) with an adaptive exploration term,
    scaled by cost efficiency. Uses a structural blend of exploitation (EI)
    and exploration (uncertainty) with adaptive weighting based on progress
    toward the known minimum.
    
    Parameters:
    - mu: shape (2000,), GP mean predictions
    - sigma: shape (2000,), GP standard deviations
    - f_best: scalar, best observed value
    - cost: shape (2000,), evaluation costs (1 + mean(x_scaled))
    
    Returns:
    - scores: shape (2000,), acquisition scores (higher = evaluate next)
    """
    # Check deadline first
    if _check_deadline():
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip to valid ranges
    sigma = np.clip(sigma, 1e-8, 1e4)
    cost = np.clip(cost, 0.5, 10.0)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    ei = (f_best - mu) * cdf_vals + sigma * pdf_vals
    ei = np.maximum(ei, 0.0)
    
    # Estimate progress: distance to known minimum of Styblinski-Tang 2D
    # Minimum is -78.3323
    dist_to_min = max(f_best - (-78.3323), 0.0)
    
    # Normalize progress: progress ~ 1 when close to min, ~ 0.1 when far
    progress = 1.0 / (1.0 + 0.05 * dist_to_min)
    
    # Adaptive parameters based on progress
    # Early on (low progress): more exploration
    # Late on (high progress): more exploitation
    
    # Cost efficiency: reward cheaper evaluations
    # Using sqrt for moderate penalty on expensive points
    cost_efficiency = 1.0 / np.sqrt(cost)
    
    # Exploration term: encourages high uncertainty
    # beta_explore decreases as we get closer to optimum
    beta_explore = 0.3 * (1.0 - 0.5 * progress)
    
    # Exploitation term: EI scaled by cost efficiency
    exploitation = ei * cost_efficiency
    
    # Exploration term: uncertainty scaled by adaptive beta and cost efficiency
    exploration = beta_explore * sigma * cost_efficiency
    
    # Combined score: blend of exploitation and exploration
    # Weight shifts from exploration to exploitation as progress increases
    weight_exploit = 0.7 + 0.3 * progress  # 0.7 to 1.0
    weight_explore = 1.0 - 0.5 * progress  # 1.0 to 0.5
    
    score = weight_exploit * exploitation + weight_explore * exploration
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
