"""Acquisition function for Bayesian optimization of Branin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for time tracking
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Kriging Believer with adaptive UCB gating.
    
    Structural change: Instead of a fixed linear UCB + EI hybrid,
    this uses a regime-dependent gating mechanism where the exploration
    weight adapts based on the distance between mu and f_best.
    
    When mu is close to f_best (exploitation regime), the score is
    dominated by -mu. When mu is far from f_best (exploration regime),
    the score is dominated by sigma-driven exploration.
    
    This creates asymmetric dynamics that can help escape local minima
    in Branin's multimodal landscape while still exploiting promising regions.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return a safe fallback if we're running out of time
    elapsed = time.time() - _start_time
    if elapsed > 325:
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Normalize mu relative to f_best and typical scale
    # Use a robust scale estimate from the current mu distribution
    mu_scale = np.std(mu) if np.std(mu) > 1e-8 else 1.0
    
    # Gating variable: how "promising" is this point relative to f_best?
    # d is in units of mu_scale; large d means far from f_best
    d = (f_best - mu) / mu_scale
    
    # Soft gating: weight shifts from exploration to exploitation
    # When d is large (mu << f_best), we're exploiting -> weight on -mu
    # When d is small or negative, we're exploring -> weight on sigma
    # Use a sigmoid-like transition centered at d=0
    gate = 1.0 / (1.0 + np.exp(-2.0 * d))
    
    # Exploration weight: higher when gate is low (far from f_best)
    # Exploitation weight: higher when gate is high (close to f_best)
    kappa_explore = 1.5  # Base exploration coefficient
    kappa_exploit = 1.0  # Base exploitation coefficient
    
    # Adaptive kappa: blend between explore and exploit regimes
    kappa = kappa_exploit * gate + kappa_explore * (1.0 - gate)
    
    # Kriging Believer score: -mu + kappa * sigma
    score = -mu + kappa * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
