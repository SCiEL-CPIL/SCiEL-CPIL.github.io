"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: must return well before the 330s wall-clock kill.
_DEADLINE = time.time() + 300.0  # 300s margin
_DEADLINE_HARD = time.time() + 325.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization on rosenbrock 2d.
    
    Parameters
    ----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP std dev predictions
    f_best : float - lowest observed value
    cost : array of shape (2000,) - evaluation costs
    
    Returns
    -------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    start = time.time()
    
    # Basic safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z for EI
    diff = f_best - mu
    z = diff / sigma_safe
    z = np.clip(z, -50, 50)
    
    # Expected Improvement (EI)
    ei = diff * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB with adaptive kappa
    # Use a higher kappa for better exploration, especially early on
    kappa = 1.5
    ucb = mu + kappa * sigma_safe
    
    # Cost-aware versions
    # For minimization, we want low mu and high sigma (exploration)
    # EI is already oriented for minimization (higher EI = better)
    # UCB needs to be negated for minimization: we want to maximize -mu + kappa*sigma
    ucb_min = -mu + kappa * sigma_safe
    
    # Normalize scores to comparable scales
    # EI is in objective units, UCB is in objective units
    # Cost divides them
    
    # Weighted blend: 85% EI, 15% UCB (from inspiration that improved to 1.955)
    w_ei = 0.85
    w_ucb = 0.15
    
    score_ei = ei / cost_safe
    score_ucb = ucb_min / cost_safe
    
    # Combine
    score = w_ei * score_ei + w_ucb * score_ucb
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Check deadline
    if time.time() > _DEADLINE:
        return score
    
    return score
# EVOLVE-BLOCK-END
