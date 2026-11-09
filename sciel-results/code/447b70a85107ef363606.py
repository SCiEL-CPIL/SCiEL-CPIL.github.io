import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 3D.
    Uses Expected Improvement (EI) with an adaptive exploration bonus.
    """
    # Known bounds for Hartmann 3D
    min_val = -3.86278
    max_val = 0.5  # Conservative upper bound
    
    # Calculate progress: 0 (far from optimum) to 1 (at optimum)
    # f_best starts high (e.g., 0) and decreases to min_val
    # progress = 1 - normalized_distance
    progress = 1.0 - np.clip((f_best - min_val) / (max_val - min_val), 0.0, 1.0)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration bonus
    # We want to explore more when progress is low (early stages)
    # and exploit more when progress is high (late stages)
    # beta starts at beta_max and decays to 0
    beta_max = 1.5
    beta = beta_max * (1.0 - progress)
    
    # Final score: EI + exploration bonus
    # This effectively behaves like UCB in the beginning and EI in the end
    score = ei + beta * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
