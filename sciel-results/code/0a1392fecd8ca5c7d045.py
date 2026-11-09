"""Refined EI + UCB blend with adaptive exploration and robust normalization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def _robust_normalize(x, clip_pct=1.0):
    """
    Normalize an array to [0, 1] using percentile clipping for robustness.
    Clips extreme values at the clip_pct-th and (100-clip_pct)-th percentiles
    before min-max scaling, preventing outliers from dominating the range.
    """
    if x.size == 0:
        return x.copy()
    
    p_low = np.percentile(x, clip_pct)
    p_high = np.percentile(x, 100.0 - clip_pct)
    
    if p_high - p_low < 1e-12:
        return np.zeros_like(x)
    
    x_clipped = np.clip(x, p_low, p_high)
    return (x_clipped - p_low) / (p_high - p_low)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: adaptive blend of Expected Improvement (EI) and
    a Lower Confidence Bound (LCB) exploration term for minimization.
    
    The blend weight and exploration coefficient beta are adapted based on
    the current uncertainty relative to the best observed value, allowing
    the strategy to shift from exploration-heavy to exploitation-heavy as
    the search converges.
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far (lower is better)
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    N = len(mu)
    
    # Ensure sigma is strictly positive and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    sigma_safe = np.clip(sigma_safe, 1e-10, 1e6)
    
    # --- Expected Improvement for minimization ---
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    # EI should be >= 0 for minimization; clip negatives to 0
    ei = np.maximum(ei, 0.0)
    
    # --- Exploration term: LCB = mu - beta * sigma ---
    # For minimization, lower mu and higher sigma are desirable for exploration.
    
    # Adaptive beta: more exploration when uncertainty is high relative to scale
    # Estimate the "scale" of the problem from the posterior
    mu_scale = np.std(mu) + 1e-10
    sigma_mean = np.mean(sigma_safe)
    
    # Ratio of exploration uncertainty to mean uncertainty
    # When sigma_mean is large relative to mu_scale, we're in early exploration
    uncertainty_ratio = sigma_mean / (mu_scale + sigma_mean + 1e-10)
    
    # Beta ranges from ~1.0 (more exploration) to ~3.0 (more exploitation-focused LCB)
    # Actually, for LCB, larger beta means more exploration.
    # Let's think: LCB = mu - beta*sigma. Larger beta → lower LCB for high-sigma points → more exploration.
    # We want more exploration early. uncertainty_ratio is high early.
    # So beta should be larger when uncertainty_ratio is high.
    beta = 1.5 + 2.0 * uncertainty_ratio  # ranges from 1.5 to 3.5
    
    lcb = mu - beta * sigma_safe
    
    # --- Normalize both components robustly ---
    ei_norm = _robust_normalize(ei, clip_pct=1.0)
    lcb_norm = _robust_normalize(lcb, clip_pct=1.0)
    
    # --- Adaptive blending ---
    # More exploration early (high uncertainty_ratio), more exploitation later
    # w_explore goes from ~0.4 to ~0.15
    w_explore = 0.4 - 0.25 * (1.0 - uncertainty_ratio)
    w_exploit = 1.0 - w_explore
    
    scores = w_exploit * ei_norm + w_explore * lcb_norm
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure scores are in a reasonable range
    scores = np.clip(scores, 0.0, 1.0)
    
    return scores
# EVOLVE-BLOCK-END
