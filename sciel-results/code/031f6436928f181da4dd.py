# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a single principled metric:
    Expected Improvement per unit cost, with an exploration bonus that
    scales with uncertainty and naturally decays as f_best approaches 0.
    
    The score is: EI(x) / cost(x) + kappa * sigma(x) / cost(x)
    where kappa is chosen so the exploration term is meaningful early
    but negligible when f_best is small.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - higher values are preferred
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 295.0

    # Ensure arrays are 1D and float
    mu = np.asarray(mu, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    cost = np.asarray(cost, dtype=np.float64).flatten()
    
    # Safety: avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)] where f(x) ~ N(mu, sigma^2)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    ei_per_cost = ei / cost_safe
    
    # Exploration bonus: sigma / cost, scaled by a factor that depends on f_best.
    # When f_best is large (early), we want more exploration.
    # When f_best is near 0 (late), exploration should diminish.
    # Use kappa = max(0.1, 0.5 * f_best) so it adapts to the scale of the problem.
    f_best_scale = max(abs(f_best), 1e-3)
    kappa = 0.3 * f_best_scale
    
    exploration = kappa * sigma_safe / cost_safe
    
    # Combine
    scores = ei_per_cost + exploration
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
