"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0

def _time_check():
    """Check if we're close to the deadline."""
    return (time.time() - _START_TIME) > (_DEADLINE - _MARGIN)


def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition function for Bayesian optimization of Ackley 2D.
    
    Blends UCB (exploration) and EI (exploitation) with weights that shift
    from exploration-heavy early on to exploitation-heavy as f_best approaches 0.
    
    mu, sigma: shape (2000,), posterior mean and std in objective units.
    f_best: lowest observed value.
    
    Returns 2000 finite scores, higher means evaluate next.
    """
    # Deadline guard: return simple fallback if running out of time
    if _time_check():
        return 0.5 * sigma - mu

    n = len(mu)

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, 0.0)

    # --- Expected Improvement for minimization ---
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z), z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # --- Adaptive UCB ---
    # Beta decays smoothly as f_best approaches 0 (the Ackley minimum).
    # When f_best is large (~5+), beta ~ 2.0 (heavy exploration).
    # When f_best is small (<0.1), beta ~ 0.3 (heavy exploitation).
    beta = 0.3 + 1.7 * np.exp(-f_best / 2.0)
    UCB = beta * sigma - mu

    # --- Smooth blend based on local uncertainty ---
    # Points with high sigma get more UCB weight; low sigma get more EI weight.
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min + 1e-12
    sigma_norm = (sigma - sigma_min) / sigma_range

    # Sigmoid blend: sharpness controls transition
    blend_temp = 1.0
    w_ucb = 1.0 / (1.0 + np.exp(-blend_temp * (sigma_norm - 0.5)))

    # Additionally, reduce overall UCB weight as f_best gets small
    # (global exploitation shift)
    global_exploit_weight = 1.0 / (1.0 + np.exp(-5.0 * (f_best - 0.5)))
    # When f_best >> 0.5, global_exploit_weight ~ 1 (explore)
    # When f_best << 0.5, global_exploit_weight ~ 0 (exploit)

    # Final blend: modulate the local UCB weight by the global exploration level
    w_ucb_final = w_ucb * global_exploit_weight

    scores = w_ucb_final * UCB + (1.0 - w_ucb_final) * EI

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
