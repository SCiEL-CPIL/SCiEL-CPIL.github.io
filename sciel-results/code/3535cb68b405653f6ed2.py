# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm, t as t_dist

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0 - 10.0  # safety margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Student-t based Expected Improvement
    with adaptive exploration/exploitation balance.
    
    Parameters:
    -----------
    mu : ndarray, shape (N,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (N,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (N,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (N,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a simple fallback: prefer low mu, low cost
        return -mu / (cost + 1e-9)
    
    N = len(mu)
    
    # Ensure numerical safety
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute improvement
    improvement = f_best - mu
    
    # Standardized improvement
    z = improvement / sigma_safe
    
    # Standard Normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (EI)
    ei = improvement * cdf_z + sigma_safe * pdf_z
    
    # Clip negative EI to zero (EI should be non-negative)
    ei = np.maximum(ei, 0.0)
    
    # Student-t based adjustment for model uncertainty
    # Use degrees of freedom that adapt to the number of observations
    # As we collect more data, df increases, approaching normal
    # Here we use a heuristic: df starts low (more exploration) and grows
    n_obs_estimate = max(5, int(np.sum(sigma < 0.1 * np.std(mu + sigma_safe))))
    df = np.clip(n_obs_estimate + 2, 5, 100)
    
    # Student-t CDF and PDF for the same z
    cdf_t = t_dist.cdf(z, df)
    pdf_t = t_dist.pdf(z, df)
    
    # Student-t based EI
    ei_t = improvement * cdf_t + sigma_safe * pdf_t * (df / (df - 2)) if df > 2 else ei
    
    # Blend standard EI and Student-t EI
    # Early on, trust Student-t more (more robust to model uncertainty)
    # Later, trust standard EI more
    blend_weight = np.clip(n_obs_estimate / 50.0, 0.3, 0.7)
    ei_blended = blend_weight * ei + (1 - blend_weight) * np.maximum(ei_t, 0.0)
    
    # Adaptive cost weighting
    # Use a power law that adapts to the budget consumed
    # Early: less penalty for cost (explore)
    # Late: more penalty for cost (exploit)
    cost_mean = np.mean(cost_safe)
    cost_ratio = cost_safe / (cost_mean + 1e-9)
    
    # Alpha adapts: start low (less cost-sensitive), increase over time
    # Estimate progress from sigma levels
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe) + 1e-9
    progress = 1.0 - (sigma_mean / sigma_max)
    progress = np.clip(progress, 0.0, 1.0)
    
    alpha = 0.5 + 0.8 * progress  # ranges from 0.5 to 1.3
    
    # Cost-weighted acquisition
    cost_weight = cost_safe ** (-alpha)
    
    # Combine EI with cost weight
    score = ei_blended * cost_weight
    
    # Add a small exploration bonus for high-uncertainty regions
    # Scaled down as we progress
    exploration_bonus = 0.1 * (1.0 - progress) * sigma_safe * cost_weight
    
    score = score + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Ensure non-negative
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
