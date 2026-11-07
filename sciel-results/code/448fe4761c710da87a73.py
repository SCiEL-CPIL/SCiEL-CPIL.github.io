"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_DEADLINE = 325.0  # safety margin before 330s kill
_start_time = time.time()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Gaussian Expected Improvement with a mild exploration bonus.
    
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x).
    
    A small exploration bonus (0.05 * sigma) is added to encourage coverage
    in the early stages when only 5 initial points are available in 6D space.
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Return scores that favor points with high uncertainty (exploration)
        # to at least return finite values
        return sigma.copy()
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -50, 50)
    
    # Standard Gaussian Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Add a small exploration bonus proportional to uncertainty
    # This helps prevent premature convergence in the limited-budget regime
    exploration_bonus = 0.05 * sigma_safe
    
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
