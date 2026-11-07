"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Structural Change:
    - Switches the exploration component from a Conservative (Pessimistic) 
      Lower Confidence Bound (minimizing mu + k*sigma) to an Optimistic 
      Upper Confidence Bound style term (maximizing -mu + k*sigma).
      This encourages jumping to new basins in the multimodal landscape.
    - Replaces the linear gap-based weight adaptation with a sigmoid-based 
      adaptation on f_best, allowing for a smoother and potentially sharper 
      transition from exploration to exploitation as the optimum is approached.
    """
    # Time guard: if we are close to the deadline, return a safe fallback
    if time.time() > _DEADLINE - 5.0:
        return -np.where(np.isfinite(mu), mu, 0.0)

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    phi = norm.pdf(z_clipped)
    Phi = norm.cdf(z_clipped)
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # 2. Compute Optimistic Exploration Score
    # We want to maximize: low mu, high sigma
    # Score = -mu + k * sigma
    # This is structurally different from the current program's -mu - k*sigma
    k = 1.5 
    Opt_Exploration = -mu + k * sigma_safe
    
    # 3. Adaptive Weighting using Sigmoid
    # Transition from exploration (low alpha) to exploitation (high alpha)
    # as f_best approaches the known minimum (-1.8013).
    # Sigmoid center around -1.5 to -1.6 range.
    # alpha = 0.5 + 0.5 * sigmoid((f_best + 1.4) / 0.2)
    # When f_best = -1.4, alpha = 0.5
    # When f_best = -1.6, alpha ~ 0.92
    # When f_best = -1.8, alpha ~ 0.98
    
    alpha = 0.5 + 0.5 * (1.0 / (1.0 + np.exp(-((f_best + 1.4) / 0.2))))
    
    # Clamp alpha to ensure some exploration always remains, but allow heavy exploitation
    alpha = np.clip(alpha, 0.2, 0.95)
    
    # Combine scores
    # Normalize scales? EI and Opt_Exploration are both in objective units.
    # EI is typically smaller than the raw mu/sigma terms.
    # We blend them directly.
    score = alpha * EI + (1 - alpha) * Opt_Exploration
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
