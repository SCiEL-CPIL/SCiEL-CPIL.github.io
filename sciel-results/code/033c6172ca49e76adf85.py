"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Hybrid acquisition: Expected Improvement with a tuned exploration bonus.
    # For minimization, EI encourages points with low mean and moderate uncertainty.
    # Adding a small kappa*sigma term helps escape local optima in multimodal landscapes.
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard expected improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus to help escape local optima
    # The coefficient is chosen to be small enough not to dominate EI
    # but large enough to provide meaningful exploration
    kappa = 0.1
    exploration_bonus = kappa * sigma_safe
    
    # Combine EI and exploration bonus
    score = ei + exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
