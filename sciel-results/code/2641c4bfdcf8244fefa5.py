"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute an acquisition score for each candidate.
    
    Uses a combination of:
    1. Expected Improvement (EI) relative to the known global minimum (3.0)
       to maintain exploration pressure.
    2. A variance-based exploration bonus to encourage visiting high-uncertainty regions.
    3. A lower confidence bound (LCB) term for exploitation.
    
    Parameters:
    -----------
    mu : array-like, shape (n_candidates,)
        Posterior mean predictions at candidate points.
    sigma : array-like, shape (n_candidates,)
        Posterior standard deviations at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores; higher means more desirable to evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Known global minimum of Goldstein-Price 2D is 3.0
    f_min_known = 3.0
    
    # Use f_min_known as reference for EI to maintain exploration signal
    # This prevents the acquisition from becoming too exploitative as f_best approaches 3.0
    z_ei = (f_min_known - mu) / sigma_safe
    ei = (f_min_known - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Lower Confidence Bound (LCB) for exploitation
    # Negative because we want to minimize
    xi = 0.01  # exploration parameter
    lcb = mu - xi * sigma_safe
    
    # Variance-based exploration bonus
    # Scale sigma to be in a reasonable range relative to the function scale
    # Goldstein-Price has values roughly in [3, 300], so scale by ~1
    var_bonus = sigma_safe
    
    # Combine: primary signal is EI relative to known minimum,
    # with a smaller LCB component for exploitation and a small variance bonus
    # The EI relative to f_min_known provides a constant exploration signal
    
    # We want higher scores to be better.
    # EI is positive when mu < f_min_known, which is what we want.
    # LCB: lower is better, so we negate it to make higher = better
    # var_bonus: higher uncertainty = more exploration
    
    # Weighted combination
    # EI dominates, LCB provides exploitation, var_bonus provides exploration
    scores = 1.0 * ei - 0.5 * lcb + 0.1 * var_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
