"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_TIMEOUT = 290  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best):
    """
    Acquisition function that blends Expected Improvement (EI) with Upper Confidence Bound (UCB).
    
    This structural change replaces the fixed exploration bonus of the previous version
    with a dynamic, point-wise blend of EI and UCB. The blend weights are determined by
    the local uncertainty (sigma) relative to the global uncertainty, allowing the
    algorithm to adaptively explore high-uncertainty regions while exploiting low-uncertainty
    regions where the mean is promising.
    
    Parameters:
    -----------
    mu : ndarray of shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray of shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed value so far.
    
    Returns:
    --------
    scores : ndarray of shape (2000,)
        Acquisition scores; higher values indicate more desirable points.
    """
    # Deadline guard: if we are close to the timeout, return a fast fallback
    # Pure EI is usually fast enough, but if we are in trouble, we can even
    # return a simple UCB which involves fewer special function calls.
    if time.time() - _START_TIME > _TIMEOUT:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    # For minimization, EI is the expected amount by which f_best is improved.
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Compute Upper Confidence Bound (UCB) score
    # For minimization, we want low mu and high sigma.
    # UCB_value = mu - beta * sigma.
    # We want a "score" where higher is better, so we negate the value:
    # UCB_score = -mu + beta * sigma.
    
    # Adaptive beta:
    # We want to explore more when the overall uncertainty is high relative to the spread of means.
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = max(mu_max - mu_min, 1e-9)
    sigma_mean = np.mean(sigma_safe)
    
    # exploration_factor is high if sigma is large relative to the range of mu.
    # Clip to a reasonable range to prevent extreme behavior.
    exploration_factor = np.clip(sigma_mean / (mu_range + 1e-9), 0.5, 2.0)
    beta = 1.0 + exploration_factor  # beta between 1.5 and 3.0
    
    ucb_score = -mu + beta * sigma_safe
    
    # 3. Normalize EI and UCB scores to [0, 1] range
    # This allows them to be blended meaningfully regardless of their absolute scales.
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = max(ei_max - ei_min, 1e-9)
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = max(ucb_max - ucb_min, 1e-9)
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # 4. Adaptive Blending
    # The weight of UCB (exploration) should be higher for points with higher sigma.
    # The weight of EI (exploitation) should be higher for points with lower sigma.
    
    # Normalize local sigma to [0, 1] relative to the mean sigma
    # sigma_weight is 1.0 for points with sigma == sigma_mean, 0.0 for sigma == 0
    sigma_weight = np.clip(sigma_safe / (sigma_mean + 1e-9), 0.0, 1.0)
    
    # Blend coefficient alpha:
    # alpha is the weight of EI. (1-alpha) is the weight of UCB.
    # When sigma is high (sigma_weight ~ 1), we want more UCB (lower alpha).
    # When sigma is low (sigma_weight ~ 0), we want more EI (higher alpha).
    
    # Base alpha is 0.5 (balanced).
    # We modulate it: high sigma -> alpha decreases (more UCB).
    # Low sigma -> alpha increases (more EI).
    # Let's map sigma_weight [0,1] to alpha [0.7, 0.3]
    # alpha = 0.7 - 0.4 * sigma_weight
    alpha = 0.7 - 0.4 * sigma_weight
    
    # Ensure alpha is within valid bounds
    alpha = np.clip(alpha, 0.1, 0.9)
    
    # Final score
    scores = alpha * ei_norm + (1 - alpha) * ucb_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
