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
    
    # Adaptive kappa: increase exploration as f_best is far from 0,
    # decrease as we approach the optimum (f_best -> 0)
    # f_best starts around ~1-2 for Ackley 2D, goes to 0 at optimum
    # Use a smooth transition: kappa = 1.0 + 0.5 * exp(-f_best)
    # When f_best is large (early), kappa ~ 1.5 (more exploration)
    # When f_best is small (late), kappa ~ 1.0 (less exploration, more exploitation)
    kappa = 1.0 + 0.5 * np.exp(-max(f_best, 0.0))
    
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
    
    # Adaptive alpha/beta: shift toward exploitation as f_best approaches 0
    # Early: alpha=0.85, beta=0.15 (more exploration)
    # Late: alpha=0.7, beta=0.3 (more exploitation)
    progress = 1.0 - np.exp(-max(f_best, 0.0))  # 0 when f_best=0, approaches 1 when f_best large
    alpha = 0.7 + 0.15 * progress  # 0.7 (late) to 0.85 (early)
    beta = 1.0 - alpha
    
    score = alpha * ucb_score + beta * ei_scores
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return score
        
    return score
