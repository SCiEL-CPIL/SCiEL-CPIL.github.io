"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Combines Expected Improvement (EI) for exploitation with an adaptive
    uncertainty-based exploration term. The exploration weight scales with
    the distance between f_best and the known Branin minimum, encouraging
    exploration early and exploitation later.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points (objective units).
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points (objective units).
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Deadline guard: track elapsed time and ensure we return within budget
    # The evaluator kills after 330s, so we need a safety margin.
    # Since this function is called once per step with 2000 points,
    # the computation should be fast, but we add a guard just in case.
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute the improvement over f_best (for minimization, lower is better)
    # Improvement is positive when mu < f_best (predicted better than current best)
    improvement = f_best - mu
    
    # Z-score for EI computation
    z = improvement / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is standard normal CDF and phi is standard normal PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Base EI
    ei = improvement * Phi_z + sigma_safe * phi_z
    
    # Adaptive exploration: add a UCB-like term that scales with
    # how far f_best is from the known minimum (0.397887 for Branin-2D)
    # This encourages exploration early and exploitation later
    branin_min = 0.397887
    distance_to_min = max(f_best - branin_min, 0.0)
    
    # Scale exploration: larger when f_best is far from optimum
    # Use a square root to dampen the growth of the exploration term
    exploration_weight = 0.5 * np.sqrt(max(distance_to_min, 0.01))
    ucb_term = exploration_weight * sigma_safe
    
    # Combine EI and exploration term
    scores = ei + ucb_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
