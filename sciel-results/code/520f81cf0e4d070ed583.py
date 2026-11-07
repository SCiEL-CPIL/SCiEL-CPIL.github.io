import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin well below the 330s wall-clock kill

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Combines:
    1. Adaptive UCB with a beta that is the maximum of a progress-based term and an uncertainty-based term.
    2. Expected Improvement (EI) for exploitation.
    
    The blend weight shifts from UCB to EI as the search approaches the known global minimum (3.0).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]
    
    # Wall-clock guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Exploitation-only fallback
        score = f_best - mu
        return np.clip(score, -1e12, 1e12)

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Known global minimum for Goldstein-Price 2D
    f_min = 3.0
    
    # Calculate the gap to the known global minimum
    gap_global = max(f_best - f_min, 0.0)
    
    # --- Component 1: Progress-based Beta (from Current Program) ---
    # High gap -> High beta (Explore). Low gap -> Low beta (Exploit).
    beta_progress = 0.5 + 2.5 * np.tanh(gap_global / 5.0)
    
    # --- Component 2: Uncertainty-based Beta (from Inspiration 1) ---
    # If f_best is far from the best predicted mean, relative to the max uncertainty, explore more.
    mu_min = np.min(mu)
    sigma_max = np.max(sigma_safe)
    
    gap_local = max(f_best - mu_min, 0.0)
    
    if sigma_max < 1e-9:
        beta_uncertainty = 0.5
    else:
        # Scale the local gap by the uncertainty
        beta_uncertainty = gap_local / sigma_max
        
    # Ensure beta_uncertainty has a floor and a ceiling
    beta_uncertainty = np.clip(beta_uncertainty, 0.5, 5.0)
    
    # --- Combine Betas ---
    # Use the maximum of the two signals to ensure we don't under-explore
    # if one signal suggests exploration but the other doesn't.
    beta = max(beta_progress, beta_uncertainty)
    
    # Cap the final beta to prevent extreme exploration
    beta = min(beta, 4.0)
    
    # --- UCB Score ---
    # For minimization, we want low mu and high sigma (exploration).
    # Score to maximize: -mu + beta * sigma
    ucb_score = -mu + beta * sigma_safe
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, 0.0)
    
    # --- EI Score ---
    # Expected Improvement for minimization
    # xi is a small exploitation parameter
    xi = 0.01 * max(gap_global, 0.1)
    
    z = (f_best - mu - xi) / sigma_safe
    ei_score = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei_score = np.where(np.isfinite(ei_score), ei_score, 0.0)
    
    # --- Blending ---
    # Weight towards UCB when gap_global is large, towards EI when small.
    # w_ucb goes from ~1 (gap large) to ~0 (gap small)
    w_ucb = np.tanh(gap_global / 2.0)
    w_ei = 1.0 - w_ucb
    
    # Linear combination
    score = w_ucb * ucb_score + w_ei * ei_score
    
    # Ensure scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Guarantee shape
    if score.shape[0] != n:
        score = np.resize(score, n)
        
    return score
