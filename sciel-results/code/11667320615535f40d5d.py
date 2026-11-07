import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for Bayesian optimization on Styblinski-Tang 2D.
    
    Uses a hybrid of Expected Improvement and Probability of Improvement with
    cost-aware weighting to balance exploration and exploitation while accounting
    for evaluation costs.
    """
    # Track elapsed time for deadline guard
    start_time = time.time()
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost-aware weighting: use a softer penalty than quadratic to allow
    # some exploration in higher-cost regions if the GP is very confident
    # Use geometric mean of EI and cost penalty
    cost_weight = 1.0 / np.sqrt(cost)
    
    # Combine EI and PI with cost weighting
    # Weight EI more heavily when uncertainty is high, PI when confident
    uncertainty_weight = np.minimum(1.0, sigma / (np.abs(f_best - mu) + 1e-9))
    ei_weight = 0.7 + 0.3 * uncertainty_weight
    pi_weight = 1.0 - ei_weight
    
    # Normalize EI and PI to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    pi_max = np.max(pi) if np.max(pi) > 0 else 1.0
    
    ei_normalized = ei / (ei_max + 1e-9)
    pi_normalized = pi / (pi_max + 1e-9)
    
    # Combine scores
    combined = ei_weight * ei_normalized + pi_weight * pi_normalized
    
    # Apply cost weighting
    score = combined * cost_weight
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
