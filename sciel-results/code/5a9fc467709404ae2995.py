"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement with a small adaptive exploration bonus.
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: if we're running out of time, return simple EI
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE_SECONDS:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -50, 50)
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement with numerical stability
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Adaptive exploration bonus:
    # Scale with sigma relative to the spread of (f_best - mu)
    # This encourages exploration when uncertainty is high relative to potential improvement
    delta = f_best - mu  # potential improvement (positive where mu < f_best)
    delta_scale = np.std(delta) + 1e-9
    
    # Exploration bonus: proportional to sigma, but normalized and capped
    # Use a small coefficient so it doesn't dominate EI
    exploration = 0.05 * (sigma_safe / delta_scale)
    
    # Combine: EI + small exploration bonus
    scores = ei + exploration
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
