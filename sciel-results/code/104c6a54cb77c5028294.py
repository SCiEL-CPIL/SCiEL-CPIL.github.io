"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Discrete, cost-normalized quantile-based heuristic for acquisition.
    
    Ranks candidates by the ratio of their posterior probability of exceeding
    the current best to their cost, decoupling from noisy mean estimates and
    prioritizing low-cost regions with high relative improvement likelihood.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points (in objective units).
    f_best : float
        Best observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    start_time = time.time()
    deadline = 320  # safety margin from 330s limit
    
    # Ensure sigma is positive
    sigma_safe = np.clip(sigma, 1e-10, None)
    
    # Compute z-score: probability that f(x) < f_best
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Posterior probability of improvement (exceeding current best for minimization)
    # This is the probability that a new evaluation will be better than f_best
    p_improve = norm.cdf(z)
    
    # Ensure probabilities are in [0, 1]
    p_improve = np.clip(p_improve, 1e-10, 1.0 - 1e-10)
    
    # Cost normalization: cost is in [1, 2]
    # Use inverse cost as a weight
    cost_weight = 1.0 / np.maximum(cost, 1e-10)
    
    # Base score: probability of improvement weighted by cost efficiency
    score = p_improve * cost_weight
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps navigate the Rosenbrock valley by encouraging exploration
    # when uncertainty is high relative to the scale of the problem
    uncertainty_bonus = 0.1 * (sigma_safe / np.maximum(np.abs(mu), 1e-10))
    uncertainty_bonus = np.clip(uncertainty_bonus, 0, 1.0)
    
    # Blend the quantile-based score with uncertainty bonus
    # Use a weighted combination
    score = 0.8 * score + 0.2 * uncertainty_bonus * cost_weight
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - start_time > deadline:
        pass  # The computation is already done, just return
    
    return score
# EVOLVE-BLOCK-END
