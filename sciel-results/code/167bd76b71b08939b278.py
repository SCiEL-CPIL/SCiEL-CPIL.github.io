"""Acquisition function for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimizing sphere 1d on [-5, 5].
    Uses a hybrid strategy combining Expected Improvement (EI) with
    adaptive exploration. The approach:
    
    1. Compute EI for minimization: EI = (f_best - mu) * Phi((f_best - mu)/sigma) 
       + sigma * phi((f_best - mu)/sigma)
    2. Add a small exploration bonus that scales with uncertainty
    3. Use a smooth transition between exploration and exploitation based on progress
    
    For minimization, we want to maximize the expected improvement.
    """
    # Track time to ensure we stay within budget
    t_start = time.time()
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # The sphere function f(x) = x^2 on [-5, 5] has max value 25
    f_max_possible = 25.0
    
    # Calculate progress: 0 at start (f_best ~ 25), 1 at end (f_best ~ 0)
    f_best_clipped = np.clip(f_best, 0.0, f_max_possible)
    progress = 1.0 - (f_best_clipped / f_max_possible)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # For numerical stability, use scipy's special functions if available,
    # otherwise use manual computation
    try:
        from scipy.stats import norm
        z = (f_best - mu) / sigma_safe
        # Phi(z) = norm.cdf(z), phi(z) = norm.pdf(z)
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    except ImportError:
        # Manual computation using numpy
        z = (f_best - mu) / sigma_safe
        # Standard normal PDF: phi(z) = exp(-z^2/2) / sqrt(2*pi)
        phi_z = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
        # Standard normal CDF: Phi(z) = 0.5 * (1 + erf(z/sqrt(2)))
        Phi_z = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
        ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (clip small negative values from numerical errors)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration bonus
    # Early on (low progress), we want more exploration
    # Later (high progress), we want more exploitation
    # The exploration bonus scales with sigma and decays with progress
    
    # Use a smooth decay function for the exploration coefficient
    # At progress=0: exploration_coeff = 1.0 (full exploration)
    # At progress=1: exploration_coeff = 0.1 (mostly exploitation)
    exploration_coeff = 0.1 + 0.9 * (1.0 - progress)
    
    # Exploration bonus: scale sigma by the exploration coefficient
    exploration_bonus = exploration_coeff * sigma_safe
    
    # Combine EI and exploration bonus
    # The weight on EI should increase as we make progress
    ei_weight = 0.5 + 0.5 * progress  # From 0.5 to 1.0
    exp_weight = 1.0 - ei_weight      # From 0.5 to 0.0
    
    score = ei_weight * ei + exp_weight * exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check time budget - if we're running low, return immediately
    elapsed = time.time() - t_start
    if elapsed > 280.0:  # Safety margin before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END
