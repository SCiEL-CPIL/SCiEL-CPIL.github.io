import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of styblinski tang 1d.
    
    Uses an adaptive hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB).
    The blend adapts to convergence: more exploration early, more exploitation later.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive beta: stronger exploration when uncertainty is high relative to improvement
    # Scale beta by the ratio of max sigma to typical improvement potential
    sigma_max = np.max(sigma)
    improvement_potential = np.maximum(f_best - np.min(mu), 1e-6)
    beta = np.clip(0.5 + 1.5 * (sigma_max / improvement_potential), 0.5, 3.0)
    
    # UCB: balance exploitation (-mu) and exploration (beta * sigma)
    ucb = -mu + beta * sigma
    
    # Adaptive blend weight: favor EI more as we converge (lower f_best relative to mu range)
    mu_range = np.max(mu) - np.min(mu) + 1e-9
    convergence = np.clip((f_best - np.min(mu)) / (mu_range + 1e-9), 0.0, 1.0)
    # When convergence is high (small relative improvement left), weight EI more
    weight_ei = 0.4 + 0.4 * convergence
    weight_ucb = 1.0 - weight_ei
    
    # Blend in original scale using weighted sum, then normalize for robustness
    raw_score = weight_ei * ei + weight_ucb * ucb
    
    # Normalize to [0, 1] to ensure stable ranking
    raw_min, raw_max = np.min(raw_score), np.max(raw_score)
    raw_range = raw_max - raw_min + 1e-9
    score = (raw_score - raw_min) / raw_range
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard
    if time.time() - start_time > 290:
        return -mu + 0.1 * sigma
    
    return score
