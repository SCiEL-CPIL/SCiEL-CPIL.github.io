"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses adaptive LCB (Lower Confidence Bound) with time-guard.
    LCB is structurally different from EI+LCB mix: it directly optimizes
    the trade-off between predicted minimum and uncertainty, which is
    particularly effective for multimodal functions like Weierstrass.
    """
    start_time = time.time()
    deadline = start_time + 2.0  # Safety margin well within 300s limit
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Infer exploration/exploitation phase from uncertainty relative to f_best
    sigma_mean = np.mean(sigma_safe)
    f_scale = max(abs(f_best), 1e-6)
    uncertainty_ratio = np.clip(sigma_mean / f_scale, 0.0, 2.0)
    
    # Adaptive beta: higher beta encourages more exploration
    # Early on (high uncertainty), use larger beta to explore broadly
    # Later (low uncertainty), use smaller beta to exploit
    beta = 0.5 + 1.5 * uncertainty_ratio  # Range roughly [0.5, 3.5]
    
    # LCB for minimization: mu - beta * sigma
    # Lower LCB is better (more likely to be near the minimum)
    lcb = mu - beta * sigma_safe
    
    # Check deadline before proceeding
    if time.time() - start_time > deadline * 0.5:
        # Return LCB-based scores quickly (negate so higher is better)
        scores = -lcb
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Normalize LCB to [0, 1] range for consistent scoring
    # Lower LCB is better, so we want high score for low LCB
    lcb_min, lcb_max = np.min(lcb), np.max(lcb)
    lcb_range = lcb_max - lcb_min
    if lcb_range > 1e-12:
        # Flip: high score for low LCB
        lcb_norm = (lcb_max - lcb) / lcb_range
    else:
        lcb_norm = np.ones_like(lcb)
    
    # Add a small exploitation component: prefer points where mu is already low
    # This helps refine the best region once uncertainty is low
    mu_min, mu_max = np.min(mu), np.max(mu)
    mu_range = mu_max - mu_min
    if mu_range > 1e-12:
        mu_norm = (mu_max - mu) / mu_range  # High score for low mu
    else:
        mu_norm = np.ones_like(mu)
    
    # Blend LCB and mu-based exploitation
    # When uncertainty is high, rely more on LCB (exploration)
    # When uncertainty is low, rely more on mu (exploitation)
    w_lcb = np.clip(0.6 + 0.4 * uncertainty_ratio, 0.5, 1.0)
    w_mu = 1.0 - w_lcb
    
    scores = w_lcb * lcb_norm + w_mu * mu_norm
    
    # Scale back to a meaningful magnitude
    scores = scores * lcb_range + lcb_min
    
    # Final cleanup: ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
