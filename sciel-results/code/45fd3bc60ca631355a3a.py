import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    Uses a combination of Expected Improvement (EI) and a knowledge-gradient-like
    uncertainty bonus, with adaptive exploration based on the gap between f_best and 0.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Use stable computation
    ei = np.zeros_like(mu)
    mask = z > 0
    ei[mask] = (f_best - mu[mask]) * norm.cdf(z[mask]) + sigma[mask] * norm.pdf(z[mask])
    # For z <= 0, EI is small but positive; compute directly
    ei[~mask] = (f_best - mu[~mask]) * norm.cdf(z[~mask]) + sigma[~mask] * norm.pdf(z[~mask])
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0)
    
    # Adaptive exploration: when f_best is far from 0 (global min), explore more
    # The Beale function has minimum 0 at (3, 0.5)
    # f_best starts high and decreases
    # Use a knowledge-gradient-like term: prioritize points with high uncertainty
    # that could potentially contain the global minimum
    
    # Compute a "potential" score that balances exploitation and exploration
    # The idea: if f_best is still large, we need to explore broadly
    # If f_best is small, we can focus on exploitation
    
    # Normalize sigma to [0, 1] range for the bonus term
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min
    if sigma_range > 1e-10:
        sigma_norm = (sigma - sigma_min) / sigma_range
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Adaptive beta: start high for exploration, decrease as f_best improves
    # f_best for Beale starts around 100+ and goes to 0
    # Use a logarithmic scaling
    f_best_abs = abs(f_best)
    if f_best_abs > 1e-6:
        # High exploration when f_best is large
        beta = 2.0 + 3.0 * np.exp(-0.1 * f_best_abs)
    else:
        beta = 2.0
    
    # UCB-like term for minimization
    ucb = -mu + beta * sigma
    
    # Normalize EI and UCB to comparable scales
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-10:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-10:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Weighted combination with adaptive weights
    # When f_best is large, weight exploration more
    if f_best_abs > 10:
        w_ei = 0.4
        w_ucb = 0.6
    elif f_best_abs > 1:
        w_ei = 0.5
        w_ucb = 0.5
    else:
        w_ei = 0.7
        w_ucb = 0.3
    
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
