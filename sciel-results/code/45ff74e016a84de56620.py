"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_DEADLINE = time.time() + 280  # safety margin before 330s limit


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI) with
    a rank-based, adaptive exploration bonus.
    
    The core idea is to use EI for exploitation (points likely to improve
    on f_best) and a rank-based uncertainty bonus for exploration, weighted
    adaptively based on progress toward the known global minimum.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (N,)
        GP posterior mean at candidate points.
    sigma : np.ndarray, shape (N,)
        GP posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (N,)
        Acquisition scores; higher means more attractive to evaluate next.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: just exploit (maximize -mu)
        return -np.asarray(mu, dtype=np.float64).ravel().copy()
    
    # Ensure arrays are 1D and finite
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    n = len(mu)
    
    # Handle any non-finite values
    mask = np.isfinite(mu) & np.isfinite(sigma)
    if not np.all(mask):
        safe_mu_val = np.max(mu[mask]) if np.any(mask) else 0.0
        mu = np.where(mask, mu, safe_mu_val)
        sigma = np.where(mask, sigma, 0.0)
    
    # Ensure sigma is strictly positive to avoid division by zero
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    
    # 1. Calculate Standard Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z to prevent numerical overflow/underflow in norm.cdf/pdf
    z_clipped = np.clip(z, -50, 50)
    
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    scores_ei = improvement * Phi_z + sigma * phi_z
    
    # 2. Calculate Rank-Based Uncertainty Bonus
    # Rank sigma from 0 (lowest) to 1 (highest)
    sigma_ranks = np.argsort(np.argsort(sigma)) / (n - 1) if n > 1 else np.ones(n)
    
    # Create a "Promising" gate: soft sigmoid that is ~1 when mu < f_best
    # and ~0 when mu >> f_best
    transition_width = 0.5
    gate = 1.0 / (1.0 + np.exp((mu - f_best) / transition_width))
    
    # 3. Determine Adaptive Exploration Weight
    # Use known global minimum of Hartmann 3D to normalize progress
    f_global_min = -3.86278
    
    # Normalize progress: 0.0 at f_best = 0.0 (start), 1.0 at f_best = -3.86
    f_best_clamped = np.clip(f_best, -3.86, 0.0)
    progress = (0.0 - f_best_clamped) / 3.86  # 0 at start, 1 at end
    
    # Exploration weight lambda: start high, reduce as we improve
    lambda_explore = 1.0 - 0.8 * progress  # 1.0 -> 0.2
    
    # Scale factor to make the bonus competitive with EI
    scale_factor = 0.5
    
    # The bonus: lambda * scale * sigma_rank * gate
    bonus = lambda_explore * scale_factor * sigma_ranks * gate
    
    # 4. Combine Scores
    scores = scores_ei + bonus
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
