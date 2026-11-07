import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_DEADLINE = time.time() + 280  # safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Quantile-based Exploration Acquisition Function.
    
    This function uses a structural change from the previous EI-based approach:
    it selects points in the top 5% of uncertainty (by rank) but only if their
    predicted mean is below the current best (f_best). This decouples exploration
    from the specific magnitude of GP variance, focusing instead on relative uncertainty.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP posterior mean at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher means more attractive to evaluate next.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: just exploit (maximize -mu)
        return -np.asarray(mu, dtype=np.float64).ravel().copy()
    
    # Ensure arrays are 1D and finite
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    n = len(mu)
    
    # Handle any non-finite values
    mask = np.isfinite(mu) & np.isfinite(sigma)
    if not np.all(mask):
        # Replace non-finite with worst-case for mu (high value for minimization)
        # and 0 for sigma
        safe_mu_val = np.max(mu[mask]) if np.any(mask) else 0.0
        mu = np.where(mask, mu, safe_mu_val)
        sigma = np.where(mask, sigma, 0.0)
    
    # Ensure sigma is strictly positive to avoid division by zero
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    
    # 1. Compute rank of sigma (0 = lowest uncertainty, 1 = highest)
    sigma_ranks = np.argsort(np.argsort(sigma)) / (n - 1) if n > 1 else np.ones(n)
    
    # 2. Identify top 5% of uncertainty
    top_quantile = 0.05
    threshold_rank = 1.0 - top_quantile  # 0.95
    
    # 3. Create a binary mask: True if point is in top 5% of uncertainty AND mu < f_best
    promising_mask = (sigma_ranks >= threshold_rank) & (mu < f_best)
    
    # 4. Compute scores:
    # - For promising points: high score based on how good mu is (lower mu = better)
    # - For non-promising points: low score based on mu (exploit)
    
    # Score for promising points: -mu + bonus (bonus encourages higher uncertainty)
    # Bonus scales with rank within the top 5%
    top_rank_within = (sigma_ranks - threshold_rank) / top_quantile  # 0 to 1 within top 5%
    top_rank_within = np.clip(top_rank_within, 0, 1)
    
    # Base score: -mu (higher is better for minimization)
    base_score = -mu
    
    # Bonus for promising points: encourage higher uncertainty
    bonus = 1.0 * top_rank_within  # Scale factor 1.0
    
    # Final score
    scores = base_score + np.where(promising_mask, bonus, 0.0)
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
