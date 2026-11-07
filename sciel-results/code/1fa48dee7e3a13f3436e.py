"""Cost-aware acquisition: Expected Improvement with normalized exploration for BO."""
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 325.0  # seconds, safety margin before 330s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with normalized exploration for Bayesian optimization of cosine8 8d.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs
    
    Returns:
    --------
    array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    if _remaining_time() < 3.0:
        sigma_safe = np.maximum(sigma, 1e-10)
        cost_safe = np.maximum(cost, 1e-10)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -15, 15)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = ei / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -15, 15)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI: improvement per unit cost
    ei_score = ei / cost_safe

    # Exploration term: prefer high uncertainty, penalized by cost
    # Using pure sigma rewards visiting unexplored regions, which is crucial
    # for multimodal functions like cosine8 to escape local minima.
    exp_score = sigma_safe / cost_safe

    # Normalize each component to zero-mean, unit-variance for comparable scales
    # This allows combining them with weights without one dominating due to scale
    ei_mean = np.mean(ei_score)
    ei_std = np.std(ei_score) + 1e-8
    ei_norm = (ei_score - ei_mean) / ei_std

    exp_mean = np.mean(exp_score)
    exp_std = np.std(exp_score) + 1e-8
    exp_norm = (exp_score - exp_mean) / exp_std

    # Combine: primarily exploitation, with significant exploration
    # Weights tuned to balance finding the global min vs local improvements
    w_ei = 0.75
    w_exp = 0.25
    combined = w_ei * ei_norm + w_exp * exp_norm

    # Ensure all values are finite
    combined = np.where(np.isfinite(combined), combined, 0.0)

    return combined
