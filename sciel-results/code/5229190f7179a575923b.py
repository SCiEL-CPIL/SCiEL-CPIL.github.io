"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: blends LCB (strong exploration) with Expected Improvement (exploitation).
    
    The Beale function has a narrow global minimum at (3, 0.5) and broad local minima.
    High exploration is crucial to escape local basins. We combine:
    - LCB with high beta for aggressive exploration
    - EI for exploitation when close to the optimum
    
    The blend weight adapts based on f_best: when f_best is large, we explore more;
    when f_best is small, we exploit more.
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher means evaluate next.
    """
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        # Return a simple exploitation score as fallback
        mu = np.asarray(mu, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        return -mu
    
    # Ensure inputs are numpy arrays and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace non-finite values
    finite_mu = mu[np.isfinite(mu)]
    mu_fill = np.median(finite_mu) if len(finite_mu) > 0 else 0.0
    mu = np.where(np.isfinite(mu), mu, mu_fill)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # --- LCB component (strong exploration) ---
    # High beta for aggressive exploration to escape local minima
    beta = 5.0
    lcb_scores = -mu + beta * sigma
    
    # --- EI component (exploitation) ---
    # Standard Expected Improvement for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei_scores = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # --- Adaptive blend weight ---
    # When f_best is large (early), favor exploration (LCB)
    # When f_best is small (late), favor exploitation (EI)
    # Normalize f_best relative to a scale
    # Beale function values range from 0 to ~150 in this domain
    f_scale = max(abs(f_best), 1.0)
    
    # Weight for LCB: high when f_best is large, low when f_best is small
    # Use a smooth transition: w_lcb = 1 / (1 + f_best / tau)
    # tau controls the transition point
    tau = 5.0
    w_lcb = 1.0 / (1.0 + f_scale / tau)
    w_ei = 1.0 - w_lcb
    
    # But ensure we never go below 0.3 for LCB (always keep some exploration)
    w_lcb = max(w_lcb, 0.3)
    w_ei = 1.0 - w_lcb
    
    # --- Normalize both components to comparable scales ---
    # This is important because LCB and EI have different units/scales
    lcb_mean = np.mean(lcb_scores)
    lcb_std = np.std(lcb_scores)
    if lcb_std < 1e-10:
        lcb_std = 1e-10
    lcb_norm = (lcb_scores - lcb_mean) / lcb_std
    
    ei_mean = np.mean(ei_scores)
    ei_std = np.std(ei_scores)
    if ei_std < 1e-10:
        ei_std = 1e-10
    ei_norm = (ei_scores - ei_mean) / ei_std
    
    # --- Combine ---
    scores = w_lcb * lcb_norm + w_ei * ei_norm
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
