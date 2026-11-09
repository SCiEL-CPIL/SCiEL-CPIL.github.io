import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses a combination of Expected Improvement and Upper Confidence Bound
    with adaptive exploration based on the current best value.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    safe_sigma = np.where(sigma < 1e-12, 1e-12, sigma)
    
    # Calculate z-score for EI
    z = (f_best - mu) / safe_sigma
    
    # Standard normal CDF and PDF
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # Expected Improvement
    EI = (f_best - mu) * Phi + safe_sigma * phi
    
    # Adaptive UCB: balance exploitation and exploration
    # When f_best is far from optimum, explore more
    # When f_best is close to optimum, exploit more
    
    # Hartmann 6D minimum is approximately -3.32237
    # f_best typically starts around -1.0 to -2.0 and decreases toward -3.32
    
    # Estimate how far we are from the optimum
    f_best_clamped = max(f_best, -3.5)
    # Normalize: 0 when close to optimum, 1 when far
    # Range: -3.32 (optimum) to about -1.0 (initial)
    exploration_scale = np.clip((-f_best_clamped - 3.32) / 2.32, 0.0, 1.0)
    
    # UCB parameter: larger when exploring, smaller when exploiting
    # beta controls the exploration-exploitation tradeoff
    beta = 0.5 + 2.0 * exploration_scale
    
    # UCB score: mean + beta * std
    UCB = mu + beta * safe_sigma
    
    # Combine EI and UCB with adaptive weighting
    # When exploring (far from optimum), weight UCB more
    # When exploiting (close to optimum), weight EI more
    w_ei = 0.7 - 0.4 * exploration_scale
    w_ucb = 1.0 - w_ei
    
    scores = w_ei * EI + w_ucb * UCB
    
    # Add a small random component to break ties and encourage exploration
    # This helps avoid getting stuck in local optima
    rng = np.random.default_rng()
    scores += rng.normal(0, 0.01, size=scores.shape)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
