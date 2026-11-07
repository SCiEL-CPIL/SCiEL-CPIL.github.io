"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the Gaussian process posterior.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the Gaussian process posterior.
    f_best : float
        Best observed value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # EI formula: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive cost penalty: stronger penalty when EI is low (exploration mode),
    # weaker when EI is high (exploitation mode)
    # This allows more aggressive exploration early on, but favors low-cost
    # exploitation as we converge
    
    # Normalize EI by its max to get a relative measure
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_rel = ei / ei_max
    else:
        ei_rel = np.zeros_like(ei)
    
    # Interpolate alpha between 1.0 (mild penalty) and 3.0 (strong penalty)
    # When EI is high relative to max, use weaker penalty (exploit even if costly)
    # When EI is low, use stronger penalty (must be cheap to be worth evaluating)
    alpha = 1.0 + 2.0 * (1.0 - ei_rel)
    
    score = ei / (cost_safe ** alpha)
    
    # Add uncertainty bonus scaled by cost efficiency
    # Higher uncertainty in cheap regions is more valuable
    unc_bonus = sigma_safe / (cost_safe ** 1.5)
    
    # Scale uncertainty bonus: stronger when EI is low (exploration), 
    # weaker when EI is high (exploitation)
    unc_weight = 0.05 + 0.15 * (1.0 - ei_rel)
    score = score + unc_weight * unc_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
