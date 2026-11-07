# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_DEADLINE = time.time() + 280  # safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Quantile-based exploration with exploitation gating.
    
    Uses empirical rank of uncertainty to select points in the top quantile
    of sigma, but only if their predicted mean is below the current best.
    This decouples exploration from the absolute magnitude of GP variance.
    
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
        mu = np.where(mask, mu, 0.0)
        sigma = np.where(mask, sigma, 0.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Compute rank of sigma among all candidates (0 to n-1, higher = more uncertain)
    # Use argsort twice to get ranks
    sigma_order = np.argsort(sigma)
    sigma_ranks = np.empty(n, dtype=np.float64)
    sigma_ranks[sigma_order] = np.arange(n, dtype=np.float64)
    
    # Normalize ranks to [0, 1]
    if n > 1:
        sigma_rank_norm = sigma_ranks / (n - 1)
    else:
        sigma_rank_norm = np.zeros(n)
    
    # Identify candidates in the top 5% of uncertainty
    top_quantile = 0.05
    high_uncertainty = sigma_rank_norm >= (1.0 - top_quantile)
    
    # Gate: only consider high-uncertainty points if their mean is below f_best
    # This ensures we're exploring regions where improvement is plausible
    exploitable = mu < f_best
    
    # For high-uncertainty and exploitable points, score based on expected improvement
    # For others, use a penalty based on how far below f_best their mean is
    # (i.e., -mu + some exploration bonus for high uncertainty)
    
    # Compute EI for all points (used as base score)
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei_scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # For high-uncertainty exploitable points, boost the score
    # The boost is proportional to the rank (higher rank = more uncertain = more boost)
    # This creates a smooth transition rather than a hard threshold
    boost = sigma_rank_norm * np.maximum(ei_scores, 0) * 0.5
    
    # Combine: base EI + boost for high-uncertainty exploitable points
    scores = ei_scores + boost * (high_uncertainty & exploitable)
    
    # For non-exploitable points (mu >= f_best), heavily penalize
    scores = np.where(exploitable, scores, -np.abs(mu - f_best) - 100.0)
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
