"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s. Keep a safety margin.
_DEADLINE_START = time.time()
_DEADLINE_LIMIT = 330.0  # seconds
_SAFETY_MARGIN = 5.0     # seconds

def _elapsed():
    return time.time() - _DEADLINE_START

def _remaining():
    return _DEADLINE_LIMIT - _SAFETY_MARGIN - _elapsed()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, lowest observed value
        cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    # Safety: if we're running out of time, return a simple fallback
    if _remaining() < 1.0:
        # Fallback: just use negative mu / cost as a simple heuristic
        return -mu / (cost + 1e-9)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive beta: shift from exploration to exploitation as f_best decreases
    # Early on (f_best is large), we want more exploration (higher beta)
    # Late on (f_best is small), we want more exploitation (lower beta)
    # Rosenbrock minimum is 0, so f_best approaching 0 means we're close
    # Normalize f_best to [0, 1] range for beta adaptation
    # When f_best is large (early), beta ~ 2 (more exploration)
    # When f_best is small (late), beta ~ 0.8 (more exploitation)
    
    # Use a smooth transition based on f_best
    # f_best for rosenbrock can be quite large initially, so use log scale
    f_best_abs = abs(f_best)
    
    # Transition from exploration to exploitation
    # Use a sigmoid-like function to smoothly transition beta
    # When f_best is large (>10), beta should be ~2
    # When f_best is small (<0.1), beta should be ~0.8
    # Use log(f_best_abs) as the input to the sigmoid
    if f_best_abs > 1e-6:
        log_fb = np.log(f_best_abs)
    else:
        log_fb = -14  # log(1e-6) ≈ -14
    
    # Map log_fb to a transition parameter
    # log_fb ranges from about -14 (f_best ~ 1e-6) to maybe 10 (f_best ~ 22000)
    # We want beta to go from ~2 (exploration) to ~0.8 (exploitation)
    # Use a smoothstep-like function
    # t = 0 when log_fb is small (late), t = 1 when log_fb is large (early)
    t_min, t_max = -10.0, 5.0
    t = np.clip((log_fb - t_min) / (t_max - t_min), 0.0, 1.0)
    
    # Beta: interpolate between 0.8 (exploitation) and 2.0 (exploration)
    beta = 0.8 + t * (2.0 - 0.8)
    
    # Cost-aware acquisition: EI^beta / cost
    # Using beta < 1 emphasizes exploration (small EI values get relatively higher scores)
    # Using beta > 1 emphasizes exploitation (only high EI values score well)
    
    # Compute EI^beta safely
    ei_clipped = np.maximum(ei, 1e-12)
    ei_beta = ei_clipped ** beta
    
    # Acquisition score
    scores = ei_beta / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
