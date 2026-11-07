# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 315.0  # safety margin vs 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of 1D sphere function.
    Uses Expected Improvement (EI) with a carefully tuned exploration-exploitation balance.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    fb = float(f_best)

    # Ensure sigma is positive for stable computation
    sigma_safe = np.maximum(sigma, 1e-8)

    # Compute Expected Improvement (EI)
    # EI(x) = (fb - mu) * Phi((fb - mu)/sigma) + sigma * phi((fb - mu)/sigma)
    z = (fb - mu) / sigma_safe
    ei = (fb - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Clip small negative values from numerical error
    ei = np.maximum(ei, 0.0)

    # For the sphere function in [-5, 5], the global minimum is at 0 with value 0.
    # As fb approaches 0, we want more exploitation (higher EI weight).
    # When fb is large, we want more exploration.
    
    # Use a simple, robust approach: pure EI is often very effective for 1D problems
    # where the GP is well-calibrated. The sphere function is smooth and unimodal,
    # so EI should work well.
    
    # However, to ensure some exploration early on, we can add a small UCB term
    # that decays as f_best improves.
    fb_scale = fb / (25.0 + fb)  # Maps [0, inf) to [0, 1)
    beta = 0.3 * fb_scale  # Small exploration term that vanishes as fb -> 0
    
    ucb_term = -beta * sigma_safe
    
    # Blend: EI dominates, with a tiny exploration boost early on
    score = ei + ucb_term
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # If time is running out, return immediately
    if time.time() > _DEADLINE:
        return score
    
    return score
# EVOLVE-BLOCK-END
