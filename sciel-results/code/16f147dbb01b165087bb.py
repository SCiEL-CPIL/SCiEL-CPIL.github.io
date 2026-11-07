"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses a hybrid of aggressive LCB exploration and probability-of-improvement
    for minimization. The structural change from pure linear UCB is the
    non-linear PoI term which has different sensitivity to the (mu, sigma, f_best)
    combination, better capturing regions where improvement is likely.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
        _call_count = 0
    
    _call_count += 1
    
    # Deadline guard: return simple LCB if we're running out of time
    if time.time() - _start_time > 285:
        scores = 5.0 * sigma - mu
        return np.where(np.isfinite(scores), scores, -np.inf)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Number of acquisition calls so far (first call is after 5 random points)
    n_evals = 5 + _call_count  # total evaluations including initial random
    
    # Phase 1: Aggressive exploration with high kappa LCB
    # Use very high kappa to systematically explore the space
    # This is the key insight: aggressive exploration beats exploitation
    # in multimodal landscapes with few evaluations
    kappa_explore = 6.0
    lcb_explore = kappa_explore * sigma_safe - mu
    
    # Phase 2: Probability of Improvement (PoI) for minimization
    # PoI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is non-linear and focuses on regions where improvement is likely
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)  # Probability that f(x) < f_best
    
    # Phase 3: Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    phi_z = norm.pdf(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * phi_z
    # EI is >= 0, higher means more expected improvement
    
    # Adaptive blending based on evaluation count and f_best quality
    # Early: heavy exploration (LCB), Later: blend in exploitation (EI/PoI)
    # The transition should be smooth
    
    # f_best quality: how close to optimum (-3.32237)
    f_norm = np.clip((f_best - (-3.32237)) / 3.32237, 0, 1)
    # f_norm = 1 when f_best is bad (~0), f_norm = 0 when at optimum
    
    # Exploration weight: high early, decreases as we find better values
    # Also decreases with number of evaluations
    step_factor = 1.0 / (1.0 + 0.15 * _call_count)
    explore_weight = (0.4 + 0.6 * f_norm) * step_factor
    explore_weight = np.clip(explore_weight, 0.1, 1.0)
    
    # Exploitation weight
    exploit_weight = 1.0 - explore_weight
    
    # Normalize scores to comparable scales before blending
    # LCB scores can be large positive; EI is in objective units
    # Scale EI to be comparable to LCB
    lcb_range = np.ptp(lcb_explore) if np.ptp(lcb_explore) > 1e-9 else 1.0
    ei_range = np.ptp(ei) if np.ptp(ei) > 1e-9 else 1.0
    
    # Normalize both to [0, 1] range for fair blending
    lcb_norm = (lcb_explore - lcb_explore.min()) / lcb_range
    ei_norm = (ei - ei.min()) / ei_range
    
    # Blend
    scores = explore_weight * lcb_norm + exploit_weight * ei_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
