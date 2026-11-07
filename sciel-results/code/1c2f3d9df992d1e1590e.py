import numpy as np
import time
from scipy.stats import norm

# Track state across calls within a single evaluation run
_call_count = 0
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    Combines Expected Improvement (EI) with a decaying UCB exploration term.
    
    The EI component drives exploitation toward known good regions.
    The UCB component drives exploration of uncertain regions, with beta
    decaying over the course of the optimization to shift focus from
    exploration to exploitation as more information is gathered.
    """
    global _call_count, _start_time
    
    # Initialize state on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're close to the 330s limit, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Return pure EI to finish quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei
    
    _call_count += 1
    
    # Compute Expected Improvement for minimization
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute UCB for minimization: -mu + beta * sigma
    # We want higher scores to indicate more promising points
    # For minimization, low mu is good, high sigma is interesting
    # So UCB score = -mu + beta * sigma
    
    # Adaptive beta: start high for exploration, decay to exploitation
    # 25 total evals, 5 initial, so 20 acquisition calls
    if _call_count <= 3:
        beta = 3.0
    elif _call_count <= 6:
        beta = 2.5
    elif _call_count <= 9:
        beta = 2.0
    elif _call_count <= 12:
        beta = 1.5
    elif _call_count <= 15:
        beta = 1.2
    elif _call_count <= 18:
        beta = 1.0
    else:
        beta = 0.8
    
    ucb = -mu + beta * sigma
    
    # Normalize EI and UCB to comparable scales before combining
    # This prevents one term from dominating due to scale differences
    
    # Use robust normalization: median and IQR-like scaling
    ei_median = np.median(ei)
    ei_scale = np.std(ei) + 1e-9
    ei_norm = (ei - ei_median) / ei_scale
    
    ucb_median = np.median(ucb)
    ucb_scale = np.std(ucb) + 1e-9
    ucb_norm = (ucb - ucb_median) / ucb_scale
    
    # Blend: weight EI more heavily as we progress, but keep exploration
    # Early: more exploration (higher UCB weight)
    # Late: more exploitation (higher EI weight)
    if _call_count <= 5:
        w_ei = 0.4
        w_ucb = 0.6
    elif _call_count <= 10:
        w_ei = 0.5
        w_ucb = 0.5
    elif _call_count <= 15:
        w_ei = 0.6
        w_ucb = 0.4
    else:
        w_ei = 0.7
        w_ucb = 0.3
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
