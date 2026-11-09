import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Compute Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB for exploration
    # Use a beta that adapts to the scale of sigma
    sigma_mean = np.mean(sigma)
    if sigma_mean > 1e-8:
        beta = 1.5 * sigma_mean
    else:
        beta = 1.0
    ucb = mu + beta * sigma
    
    # Blend EI and UCB
    # Weight EI more when we have good predictions, UCB when uncertain
    # Normalize both to [0, 1] range for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    if ei_max - ei_min > 1e-10:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.ones_like(ei)
    
    if ucb_max - ucb_min > 1e-10:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.ones_like(ucb)
    
    # Blend: 60% EI, 40% UCB for balanced exploration/exploitation
    score = 0.6 * ei_norm + 0.4 * (1 - ucb_norm)  # Lower UCB is better for minimization
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
