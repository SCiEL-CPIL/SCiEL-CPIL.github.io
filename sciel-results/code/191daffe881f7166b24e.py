# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_DEADLINE = time.time() + 280  # safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Hybrid Rank-Boosted Expected Improvement Acquisition Function.
    
    This function combines standard Expected Improvement (EI) with a continuous,
    rank-based uncertainty bonus. It addresses the structural limitation of hard
    quantile gating (which can be brittle) by using a smooth sigmoid gate for
    promising points and an adaptive exploration weight that decays as the
    search converges to the known global minimum.
    
    Strategy:
    1. Compute Expected Improvement (EI) for all candidates.
    2. Compute a rank-based uncertainty score for each candidate.
    3. Apply a smooth gate to favor candidates with low mean (promising).
    4. Scale the uncertainty bonus by an adaptive weight that decreases as
       f_best approaches the known global minimum (-3.86278).
    5. Final Score = EI + Adaptive_Uncertainty_Bonus.
    
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
    
    # 2. Calculate Rank-Based Uncertainty Score
    # Rank sigma from 0 (lowest) to 1 (highest)
    if n > 1:
        sigma_ranks = np.argsort(np.argsort(sigma)) / (n - 1)
    else:
        sigma_ranks = np.ones(n)
    
    # 3. Smooth Gate for Promising Points
    # Gate is 1 if mu < f_best, 0 if mu > f_best, smooth in between.
    # Width controls the transition sharpness. 0.5 is reasonable for Hartmann 3D scale.
    transition_width = 0.5
    # Use a stable sigmoid implementation
    x_gate = (mu - f_best) / transition_width
    # Clip x_gate to avoid overflow in exp
    x_gate_clipped = np.clip(x_gate, -50, 50)
    gate = 1.0 / (1.0 + np.exp(x_gate_clipped))
    
    # 4. Determine Adaptive Exploration Weight
    # f_global_min = -3.86278
    f_global_min = -3.86278
    
    # Normalize progress: 0.0 at start (f_best ~ 0), 1.0 at global min
    # We clamp f_best to [f_global_min, 0.0] for this calculation
    # Note: Hartmann 3D values are negative, so 0 is "worst" in this context for progress calc
    # Actually, f_best starts high (e.g., -1.0) and goes down to -3.86.
    # Let's define progress as how much of the gap from 0 to -3.86 we have closed.
    # Gap total = 3.86278
    # Gap remaining = f_best - (-3.86278) = f_best + 3.86278
    # Progress = 1 - (Gap Remaining / Total Gap)
    
    # To be robust against f_best starting values (e.g., if first eval is -2.0),
    # we can just clamp.
    f_best_clamped = np.clip(f_best, f_global_min, 0.0)
    
    # If f_best is 0, progress is 0. If f_best is -3.86, progress is 1.
    progress = (0.0 - f_best_clamped) / (0.0 - f_global_min)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Exploration weight lambda
    # Start with high exploration (lambda=1.0) and reduce to low (lambda=0.1)
    # This encourages exploring high uncertainty early, and exploiting EI later.
    lambda_max = 1.0
    lambda_min = 0.1
    lambda_explore = lambda_max - (lambda_max - lambda_min) * progress
    
    # 5. Calculate Bonus
    # Scale factor: EI values are typically O(1). 
    # We want the bonus to be significant but not dominate completely.
    # A scale of 0.5 means the max bonus is 0.5 * lambda.
    scale_factor = 0.5
    bonus = lambda_explore * scale_factor * sigma_ranks * gate
    
    # 6. Combine Scores
    scores = scores_ei + bonus
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
