"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import t as student_t
from scipy.stats import norm
import time

# Track cumulative time for deadline enforcement
_START_TIME = time.time()
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling with Student-t Process.
    
    Combines the structural idea of Thompson Sampling (stochastic exploration
    via sampling from the posterior) with the refinement machinery of
    Student-t Expected Improvement (heavier tails for uncertainty).
    
    The approach:
    1. Use Student-t predictive distribution (nu=3) for heavier tails.
    2. Sample from the t-distribution to get a Thompson Sampling score.
    3. Add a small EI-based exploitation term to guide toward known good regions.
    4. Adaptive balance: as f_best approaches 0, shift toward exploitation.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points (objective units, lower is better).
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points (objective units).
    f_best : float
        Best observed function value so far (lower is better).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Fallback: return negative mu (pure exploitation) to finish quickly
        return -mu
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Student-t Process parameters
    # For t-distribution with nu df, Var = nu/(nu-2) * scale^2
    # So scale = sigma * sqrt((nu-2)/nu) to match variance
    nu = 3.0
    scale_factor = np.sqrt((nu - 2.0) / nu)
    t_scale = sigma_safe * scale_factor
    
    # Thompson Sampling: sample from the Student-t predictive distribution
    # f ~ t(mu, t_scale, nu)
    # Sample: f_sample = mu + t_scale * t_sample where t_sample ~ t(0, 1, nu)
    rng = np.random.default_rng()
    t_samples = rng.standard_t(df=nu, size=mu.shape)
    sampled_values = mu + t_scale * t_samples
    
    # Thompson Sampling score: negative of sampled value (lower is better, so higher score = lower sampled value)
    ts_score = -sampled_values
    
    # Student-t Expected Improvement for minimization
    # z_t = (f_best - mu) / t_scale
    z_t = (f_best - mu) / t_scale
    
    # CDF and PDF of Student-t
    cdf_t = student_t.cdf(z_t, nu)
    pdf_t = student_t.pdf(z_t, nu)
    
    # EI for t-distribution:
    # EI = (f_best - mu) * CDF_t(z) + t_scale * nu/(nu-1) * PDF_t(z) * (1 + z^2/nu)
    correction = nu / (nu - 1.0)
    ei_t = (f_best - mu) * cdf_t + t_scale * correction * pdf_t * (1.0 + z_t**2 / nu)
    
    # Clip EI to avoid numerical issues
    ei_t = np.clip(ei_t, -1e6, 1e6)
    ei_t = np.maximum(ei_t, 0.0)
    
    # Adaptive balance between Thompson Sampling (exploration) and EI (exploitation)
    # As f_best approaches 0, we want more exploitation
    # f_best starts around 60 and should approach 0
    # Use a smooth transition: weight_ei increases as f_best decreases
    f_best_clipped = max(f_best, 1e-6)
    
    # Weight for EI: starts low (~0.2) when f_best is large, increases to ~0.8 when f_best is small
    # Use logistic-like decay: weight_ei = 1 / (1 + exp(-k * (f_ref - f_best)))
    # f_ref = 10.0, k = 0.5
    f_ref = 10.0
    k = 0.5
    weight_ei = 1.0 / (1.0 + np.exp(-k * (f_ref - f_best_clipped)))
    # Clip weight to [0.2, 0.8]
    weight_ei = np.clip(weight_ei, 0.2, 0.8)
    weight_ts = 1.0 - weight_ei
    
    # Normalize both scores to have similar scale before combining
    ts_std = np.std(ts_score)
    ei_std = np.std(ei_t)
    
    if ts_std > 1e-9:
        ts_norm = ts_score / ts_std
    else:
        ts_norm = ts_score
    
    if ei_std > 1e-9:
        ei_norm = ei_t / ei_std
    else:
        ei_norm = ei_t
    
    # Combine: weighted sum
    scores = weight_ts * ts_norm + weight_ei * ei_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
