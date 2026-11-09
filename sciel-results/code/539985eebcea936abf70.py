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
    # Use a softer inverse cost weight to avoid over-penalizing expensive points
    # that might have high potential improvement
    cost_weight = 1.0 / np.maximum(cost, 1e-10)
    
    # Expected improvement (EI) for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where phi is the standard normal PDF
    z_safe = np.clip(z, -10, 10)  # avoid numerical issues
    ei = (f_best - mu) * norm.cdf(z_safe) + sigma_safe * norm.pdf(z_safe)
    ei = np.maximum(ei, 0)  # EI should be non-negative
    
    # Normalize EI by its maximum to get a [0, 1] scale
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Blend POI and normalized EI for a more nuanced exploitation signal
    # POI gives probability, EI gives magnitude; combining them balances
    # "likely to improve" with "how much could improve"
    exploitation_score = 0.6 * p_improve + 0.4 * ei_norm
    
    # Base score: combined exploitation weighted by cost efficiency
    score = exploitation_score * cost_weight
    
    # Exploration bonus: favor regions with high uncertainty relative to the
    # current best value scale, which helps navigate the Rosenbrock valley
    # Use f_best as a natural scale reference instead of |mu|
    f_scale = max(abs(f_best), 1e-10)
    uncertainty_bonus = sigma_safe / f_scale
    uncertainty_bonus = np.clip(uncertainty_bonus, 0, 2.0)
    
    # Apply a gentle saturation to avoid excessive exploration
    # tanh-like scaling: maps [0, 2] to approximately [0, 0.96]
    uncertainty_bonus = np.tanh(uncertainty_bonus)
    
    # Blend exploitation and exploration with cost-aware weighting
    # Increase exploration weight slightly to help escape local minima
    score = 0.75 * score + 0.25 * uncertainty_bonus * cost_weight
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - start_time > deadline:
        pass  # The computation is already done, just return
    
    return score
# EVOLVE-BLOCK-END
