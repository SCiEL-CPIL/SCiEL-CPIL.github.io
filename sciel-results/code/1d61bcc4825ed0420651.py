"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of 1D sphere on [-5, 5].
    
    Uses a hybrid approach:
    - Early on (high uncertainty), use UCB-like exploration
    - Later, use Expected Improvement for exploitation
    
    Since we don't know the step number, we use the relationship between
    f_best and the scale of mu/sigma to adapt.
    
    For sphere function f(x) = x^2 on [-5, 5]:
    - f_best starts high (~25) and decreases toward 0
    - We can use f_best as a proxy for progress
    
    Strategy: Use a weighted combination of EI and UCB, with the weight
    shifting from exploration to exploitation as f_best decreases.
    """
    # Safety: ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB for minimization: prefer low mu and high sigma
    # Scale kappa based on how much we've improved
    # When f_best is large (early), explore more; when small, exploit
    # f_best ranges from ~25 to ~0 for x^2 on [-5,5]
    # Use a smooth transition
    progress = 1.0 - min(f_best / 25.0, 1.0)  # 0 early, 1 late
    kappa = 0.5 + 2.0 * (1.0 - progress)  # Start with kappa=2.5, end with 0.5
    
    ucb = mu - kappa * sigma_safe
    
    # Normalize both to comparable scales
    # EI is in objective units, UCB is in objective units
    # Blend them
    # Early: more UCB, Later: more EI
    w_ei = 0.3 + 0.7 * progress
    w_ucb = 1.0 - w_ei
    
    score = w_ei * ei + w_ucb * ucb
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
