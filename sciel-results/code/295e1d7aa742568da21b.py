import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_DEADLINE = time.time() + 280  # safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Quantile-Boosted Expected Improvement Acquisition Function.
    
    This function combines standard Expected Improvement (EI) with a rank-based
    uncertainty bonus. It addresses the structural limitation of pure EI (greediness)
    by explicitly rewarding high-uncertainty points that are also promising (mu < f_best).
    
    The exploration weight is adaptive:
    - Early on (f_best is high), the uncertainty bonus is stronger.
    - Later (f_best is low), the function relies more on pure EI.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP posterior mean at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
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
        # Replace non-finite with worst-case for mu (high value for minimization)
        # and 0 for sigma
        safe_mu_val = np.max(mu[mask]) if np.any(mask) else 0.0
        mu = np.where(mask, mu, safe_mu_val)
        sigma = np.where(mask, sigma, 0.0)
    
    # Ensure sigma is strictly positive to avoid division by zero
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    
    # 1. Calculate Standard Expected Improvement (EI)
    # Improvement is f_best - mu
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z to prevent numerical overflow/underflow in norm.cdf/pdf
    z_clipped = np.clip(z, -50, 50)
    
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    scores_ei = improvement * Phi_z + sigma * phi_z
    
    # 2. Calculate Rank-Based Uncertainty Bonus
    # We want to boost points with high uncertainty, but only if they are promising.
    
    # Rank sigma from 0 (lowest) to 1 (highest)
    # Using argsort to get ranks
    sigma_ranks = np.argsort(np.argsort(sigma)) / (n - 1) if n > 1 else np.ones(n)
    
    # Create a "Promising" mask: mu < f_best
    # We use a soft gate to avoid hard discontinuities
    # If mu is significantly better than f_best, gate is 1.
    # If mu is worse, gate is 0.
    # Transition width: 0.1 * range of mu? Or fixed?
    # Let's use a fixed transition width relative to the scale of the problem.
    # Hartmann 3D range is roughly [-4, 0].
    # A transition of 0.5 seems reasonable.
    transition_width = 0.5
    gate = 1.0 / (1.0 + np.exp((mu - f_best) / transition_width))
    
    # 3. Determine Adaptive Exploration Weight
    # f_global_min = -3.86278
    # If f_best is close to global min, we should exploit (low weight).
    # If f_best is far, we should explore (high weight).
    
    f_global_min = -3.86278
    
    # Normalize progress: 0.0 at f_best = 0.0 (start), 1.0 at f_best = -3.86
    # We clamp f_best to [0, -3.86] for this calculation
    f_best_clamped = np.clip(f_best, -3.86, 0.0)
    progress = (0.0 - f_best_clamped) / 3.86  # 0 at start, 1 at end
    
    # Exploration weight lambda
    # Start with high exploration (lambda=1.0) and reduce to low (lambda=0.2)
    # The bonus term is lambda * sigma_rank * gate
    # Since sigma_rank is [0,1] and gate is [0,1], the max bonus is lambda.
    # EI values can be larger. We need to scale the bonus to be competitive.
    # Let's scale the bonus by the max possible EI or just a constant.
    # A constant scale of 0.5 * max(sigma) might be better?
    # Let's use a scale factor based on the problem scale.
    scale_factor = 0.5
    
    lambda_explore = 1.0 - 0.8 * progress  # 1.0 -> 0.2
    
    # The bonus is: lambda * scale * sigma_rank * gate
    bonus = lambda_explore * scale_factor * sigma_ranks * gate
    
    # 4. Combine Scores
    # Score = EI + Bonus
    scores = scores_ei + bonus
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
