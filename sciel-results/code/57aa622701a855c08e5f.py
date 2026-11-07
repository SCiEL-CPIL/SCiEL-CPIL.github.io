"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more desirable to evaluate next.
    """
    t_start = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB with adaptive kappa based on confidence
    # Higher sigma relative to improvement potential -> more exploration
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # Adaptive kappa: larger when we're uncertain
    # Use a smooth function of z to balance exploration/exploitation
    # When z is very negative (mu >> f_best), we should explore less
    # When z is near 0 or positive, we can explore more
    kappa_base = 1.2
    # Scale kappa down when we're very confident (z >> 0)
    kappa = kappa_base * np.exp(-0.1 * np.maximum(z, 0))
    kappa = np.maximum(kappa, 0.3)
    
    # UCB score for minimization: f_best - mu + kappa * sigma
    ucb = f_best - mu + kappa * sigma
    
    # Expected Improvement as a secondary signal
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Combine: primarily UCB, with a small EI contribution
    # Weight EI more when sigma is small (exploit) and UCB when sigma is large (explore)
    sigma_norm = sigma / (np.max(sigma) + 1e-9)
    w_ei = 0.3 * (1.0 - sigma_norm)
    w_ucb = 1.0 - w_ei
    
    combined = w_ucb * ucb + w_ei * ei
    
    # Normalize by cost
    scores = combined / cost
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - t_start > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
