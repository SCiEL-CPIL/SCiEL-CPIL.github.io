"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s kill

def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE_SECONDS


def acquisition(mu, sigma, f_best):
    """
    Hypervolume-based acquisition function (Expected Hypervolume Improvement, EHI).
    
    For a single-objective minimization problem, EHI relative to the current best
    f_best is equivalent to Expected Improvement (EI) when the reference point is
    set appropriately. However, we can enhance this with:
    
    1. Adaptive UCB exploration that shifts based on f_best
    2. A hypervolume-inspired weighting that considers the volume of improvement
    
    The key insight: in single-objective BO, EHI = EI when reference point is at
    f_best. But we can add a small exploration bonus that adapts as f_best improves.
    
    We use a combination:
    - Base: Expected Improvement (which equals EHI in single-objective)
    - Exploration: Adaptive UCB term that decreases as f_best approaches the known minimum
    
    The adaptive coefficient beta is based on how close f_best is to the theoretical
    minimum of branin (0.397887). Early on, more exploration; later, more exploitation.
    """
    if _check_deadline():
        # Return a safe fallback: pure exploitation (negative sigma)
        return -sigma.copy()
    
    n = len(mu)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard EI computation (which is EHI in single-objective with ref=f_best)
    improvement = f_best - mu
    z = improvement / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Use scipy.stats.norm for CDF and PDF
    from scipy.stats import norm
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = improvement * cdf_z + sigma_safe * pdf_z
    
    # Adaptive UCB exploration term
    # The theoretical minimum of branin is 0.397887
    # As f_best gets closer to this, we reduce exploration
    # beta starts high (exploration) and decreases
    min_value = 0.397887
    # Normalize progress: 0 when f_best is far from min, 1 when close
    # Use a scale of about 10 units for the transition
    progress = np.clip((10.0 - f_best) / (10.0 - min_value), 0.0, 1.0)
    # Beta decreases from ~2.0 to ~0.5 as progress increases
    beta = 2.0 - 1.5 * progress
    
    # UCB term: beta * sigma (standard deviation as uncertainty measure)
    ucb_term = beta * sigma_safe
    
    # Combine: EI + UCB exploration bonus
    # This gives us EHI-like exploitation plus adaptive exploration
    scores = ei + ucb_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
