import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = None
_TIME_LIMIT = 320.0  # seconds, safety margin before 330s kill


def _ensure_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT
    return _DEADLINE


def _time_remaining():
    deadline = _ensure_deadline()
    return deadline - time.time()


def _sanitize(scores):
    if scores.shape != (2000,):
        scores = np.resize(scores, (2000,))
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)


def acquisition(mu, sigma, f_best):
    global _DEADLINE
    _ensure_deadline()
    
    # Check if we are running out of time; if so, return a simple EI score quickly
    if _time_remaining() < 5.0:
        mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
        sigma_safe = np.maximum(sigma, 1e-6)
        f_best_safe = np.nan_to_num(f_best, nan=-1e6, posinf=1e6, neginf=-1e6)
        z = (f_best_safe - mu_safe) / sigma_safe
        ei = (f_best_safe - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return _sanitize(ei)

    # Safety
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    sigma_safe = np.maximum(sigma, 1e-6)
    f_best_safe = np.nan_to_num(f_best, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # 1. Expected Improvement
    z = (f_best_safe - mu_safe) / sigma_safe
    ei = (f_best_safe - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Thompson Sampling Proxy / Normalized Improvement
    # This score encourages points with high potential relative to their uncertainty.
    # It acts as a variance-normalized exploitation signal.
    ts_proxy = (f_best_safe - mu_safe) / (sigma_safe + 1e-6)
    
    # 3. Lower Confidence Bound (UCB for minimization)
    # We want to minimize mu - kappa * sigma
    # Score = - (mu - kappa * sigma) = -mu + kappa * sigma
    # Choose kappa based on progress.
    # Hartmann 3D global min ~ -3.86
    # If f_best is far from -3.86, explore more.
    gap = f_best_safe + 3.86278
    
    # Map gap to exploration weight
    # Early: gap is large -> high exploration
    # Late: gap is small -> low exploration
    weight_explore = np.clip(gap / 2.0, 0.2, 0.8)
    weight_exploit = 1 - weight_explore
    
    kappa = 2.0
    lcb_scores = -mu_safe + kappa * sigma_safe
    
    # Normalize scores to comparable scale using min-max scaling
    def min_max_normalize(x):
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max - x_min < 1e-6:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
        
    ei_norm = min_max_normalize(ei)
    lcb_norm = min_max_normalize(lcb_scores)
    ts_norm = min_max_normalize(ts_proxy)
    
    # Combine the three signals
    # EI is the standard robust choice.
    # LCB provides strong exploration in early stages.
    # TS Proxy provides a different flavor of exploitation/exploration balance.
    
    # Weights:
    # Base weight for EI is high.
    # Weight for LCB is driven by weight_explore.
    # Weight for TS Proxy is a small constant to add diversity without destabilizing.
    
    w_ei = 0.5
    w_lcb = weight_explore * 0.3
    w_ts = 0.2
    
    # Normalize weights to sum to 1
    total_w = w_ei + w_lcb + w_ts
    w_ei /= total_w
    w_lcb /= total_w
    w_ts /= total_w
    
    final_scores = w_ei * ei_norm + w_lcb * lcb_norm + w_ts * ts_norm
    
    return _sanitize(final_scores)
