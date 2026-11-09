import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Blends Expected Improvement (EI) and Upper Confidence Bound (UCB) terms
    with adaptive weighting, normalized by a power of the evaluation cost.
    
    Parameters:
    mu (np.ndarray): Mean predictions from the GP, shape (N,)
    sigma (np.ndarray): Standard deviation predictions from the GP, shape (N,)
    f_best (float): The best (lowest) observed objective value so far.
    cost (np.ndarray): Cost of evaluating each point, shape (N,)
    
    Returns:
    np.ndarray: Acquisition scores, higher is better, shape (N,)
    """
    # Deadline guard: track elapsed time
    t_start = time.time()
    
    # Numerical stability
    eps = 1e-9
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, eps)
    
    # Check for early exit if we're running out of time
    # The acquisition function itself should be fast, but let's be safe
    if time.time() - t_start > 280:
        # Return a simple fallback
        return np.zeros_like(mu)
    
    # Exploration parameter xi for EI.
    # Ackley is multimodal, so some exploration is beneficial.
    xi = 0.05
    
    # Calculate z-score for modified EI
    z = (f_best - mu - xi * sigma_safe) / sigma_safe
    
    # Calculate CDF and PDF of standard normal distribution
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement with xi
    ei = (f_best - mu - xi * sigma_safe) * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)
    
    # UCB term: mu + kappa * sigma
    # kappa controls the exploration/exploitation tradeoff
    # Use a smaller kappa since we're already using EI
    kappa = 0.5
    ucb = mu + kappa * sigma_safe
    
    # Adaptive weighting: blend EI and UCB
    # When f_best is close to 0 (minimum), we want more exploitation (EI)
    # When f_best is far from 0, we want more exploration (UCB)
    # Use a smooth transition based on f_best
    # f_best for Ackley ranges from ~0 to ~20
    # Normalize f_best to [0, 1] for weighting
    f_best_norm = np.clip(f_best / 20.0, 0.0, 1.0)
    
    # Weight for EI: higher when f_best is small (close to optimum)
    w_ei = 1.0 - 0.5 * f_best_norm  # ranges from 0.5 to 1.0
    w_ucb = 0.5 * f_best_norm        # ranges from 0.0 to 0.5
    
    # Normalize weights
    w_sum = w_ei + w_ucb
    w_ei /= w_sum
    w_ucb /= w_sum
    
    # Normalize EI and UCB to comparable scales
    # EI is in objective units, UCB is also in objective units
    # But UCB can be negative for Ackley, so shift it
    ucb_shifted = ucb - np.min(ucb) if np.min(ucb) < 0 else ucb
    
    # Blend
    blended = w_ei * ei + w_ucb * ucb_shifted
    
    # Cost-aware scoring
    # Divide by cost^gamma. gamma < 1 makes the score less sensitive to cost.
    gamma = 0.3
    cost_safe = np.maximum(cost, eps)
    scores = blended / np.power(cost_safe, gamma)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    if time.time() - t_start > 290:
        return scores
    
    return scores
