import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Calculate progress: 0 when f_best=0 (optimal), approaches 1 when f_best is large
    # Using exponential decay to map [0, inf) to [0, 1)
    # Ackley 2D min is 0. Typical initial f_best ~ 1-2.
    progress = 1.0 - np.exp(-max(f_best, 0.0))
    
    # Adaptive kappa for UCB
    # Early (progress ~ 1): kappa ~ 1.5 (more exploration)
    # Late (progress ~ 0): kappa ~ 0.5 (less exploration, more exploitation)
    kappa = 0.5 + 1.0 * progress
    
    ucb_score = -mu + kappa * sigma
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    # Handle division by zero in sigma
    sigma_safe = np.where(sigma < 1e-10, 1.0, sigma)
    z = (f_best - mu) / sigma_safe
    
    # Calculate Phi(z) and phi(z)
    phi_z = norm.pdf(z)
    phi_cdf_z = norm.cdf(z)
    
    ei_scores = (f_best - mu) * phi_cdf_z + sigma_safe * phi_z
    
    # For points where sigma was 0 (or very small), correct the score
    # If sigma < 1e-10, the variance is negligible.
    # The improvement is simply max(0, f_best - mu)
    mask_low_sigma = sigma < 1e-10
    ei_scores[mask_low_sigma] = np.maximum(0, f_best - mu[mask_low_sigma])
    
    # Ensure finite values
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, 0.0)
    
    # Adaptive weights:
    # Early (progress ~ 1): alpha ~ 0.8 (mostly UCB)
    # Late (progress ~ 0): alpha ~ 0.4 (more balanced, leaning EI)
    alpha = 0.4 + 0.4 * progress
    beta = 1.0 - alpha
    
    score = alpha * ucb_score + beta * ei_scores
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return score
        
    return score
