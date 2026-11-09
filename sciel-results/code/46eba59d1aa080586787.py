"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with adaptive
    exploration bonus for Bayesian optimization of Beale 2D.

    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at 2000 candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at 2000 candidate points.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns:
    --------
    np.ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    start_time = time.time()

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)

    # --- Expected Improvement (for minimization) ---
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # Higher EI means the point is expected to improve upon f_best.
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid extreme values causing numerical issues
    z = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)

    # --- Adaptive exploration bonus ---
    # Early in the search (f_best far from 0), explore more.
    # Late (f_best near 0), exploit more.
    # Beale function: min=0 at (3, 0.5), max ~60.
    # Use a smooth schedule: bonus coefficient decays as f_best approaches 0.
    f_best_clipped = max(f_best, 1e-6)
    # beta ranges from ~1.2 (very exploratory) down to ~0.3 (exploitative)
    # Using a hyperbolic decay: beta = beta_min + (beta_max - beta_min) * (f_best/(f_best + tau))
    beta_min = 0.3
    beta_max = 1.2
    tau = 8.0  # characteristic value where beta is halfway
    beta = beta_min + (beta_max - beta_min) * (f_best_clipped / (f_best_clipped + tau))

    # Exploration bonus: beta * sigma (standard deviation bonus)
    # This encourages visiting high-uncertainty regions
    exploration_bonus = beta * sigma_safe

    # --- Combined score ---
    # Primary signal: EI (exploits known good regions)
    # Secondary signal: exploration bonus (explores uncertain regions)
    # Scale EI to be comparable to the exploration bonus.
    # EI is in objective units; sigma is also in objective units (standardized targets).
    # We weight EI heavily but add the bonus to encourage exploration.
    # Normalize EI by a characteristic scale so the bonus has meaningful weight.
    # A typical EI value when f_best is moderate might be ~1-5.
    ei_scale = max(f_best_clipped, 1.0) * 0.5  # adaptive scale
    ei_normalized = ei / (ei_scale + 1e-6)

    scores = ei_normalized + exploration_bonus

    # Safety: ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)

    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 280:
        return scores

    return scores
# EVOLVE-BLOCK-END
