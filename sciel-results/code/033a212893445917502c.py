"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds of safety margin


def _elapsed():
    return time.time() - _start_time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Strategy: Monte Carlo Expected Improvement (MC-EI) combined with a
    strong exploitation term (-mu).
    
    Instead of using the analytic EI formula which can suffer from numerical
    instability, we estimate the Expected Improvement via Monte Carlo sampling.
    We then combine this MC-EI score with a strong exploitation term (-mu) to
    ensure that regions with very low predicted means are prioritized, even if
    their uncertainty is low. This creates a "best-of-two" dynamic:
    1. If a region has a very low mean, -mu will be high.
    2. If a region has high potential for improvement (high EI), MC-EI will be high.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    # Deadline check
    if _elapsed() > _DEADLINE_MARGIN:
        # Return a safe fallback: just negative mu (exploitation only)
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        scores = -mu
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # --- Monte Carlo Expected Improvement ---
    # We want to estimate E[max(f_best - f, 0)] where f ~ N(mu, sigma^2)
    
    # Number of samples per point. 
    # 100 samples is a good trade-off between accuracy and speed for 2000 points.
    K = 100
    
    # Generate random samples: shape (K, n)
    # Standard normal samples
    z = np.random.standard_normal((K, n))
    
    # Transform to predicted values: f_sample = mu + sigma * z
    # mu and sigma are (n,), so we broadcast
    f_samples = mu[None, :] + sigma_safe[None, :] * z
    
    # Compute improvement: max(f_best - f_samples, 0)
    improvements = np.maximum(f_best - f_samples, 0.0)
    
    # Average over samples to get MC-EI estimate: shape (n,)
    mc_ei = np.mean(improvements, axis=0)
    
    # --- Exploitation Term ---
    # We want to reward low mu values.
    # Simple approach: -mu
    
    # Let's use a weighted sum:
    # score = w_exploit * (-mu) + w_ei * mc_ei
    
    # Adaptive weights?
    # Early on, sigma is large, so mc_ei is large. -mu might be noisy.
    # Late on, sigma is small, mc_ei is small, -mu dominates.
    # This seems naturally adaptive.
    
    # Let's try a fixed weight first to see the behavior.
    # If we use equal weights, the scales might be different.
    
    scores = -mu + mc_ei
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Final deadline check
    if _elapsed() > _DEADLINE_MARGIN:
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
