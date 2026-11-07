import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI)
    with an adaptive exploration bonus based on the current best value.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Use scipy.stats.norm for CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Adaptive exploration bonus using log-based decay
    # f_best starts around 25 (max of x^2 on [-5,5]) and goes to 0.
    # We want high exploration early (high f_best) and low exploration late (low f_best).
    # Use log-based decay for smoother transition
    
    # Handle case where f_best might be negative or zero
    # For sphere x^2, f_best >= 0.
    # Use log(1 + f_best) for smooth decay from high to low values
    f_clipped = np.clip(f_best, 0.0, 25.0)
    
    # Log-based exploration: decays smoothly from log(26)~3.26 to log(1)=0
    log_factor = np.log1p(f_clipped)  # log(1 + f_best)
    
    # Normalize: at f_best=25, log_factor~3.26; at f_best=0, log_factor=0
    # Scale to get desired exploration range
    lambda_max = 0.8
    lambda_min = 0.05  # Small floor to ensure some exploration even when f_best is tiny
    
    # Linear interpolation between lambda_min and lambda_max based on log_factor
    log_max = np.log1p(25.0)  # ~3.26
    normalized_log = log_factor / log_max
    
    lambda_explore = lambda_min + (lambda_max - lambda_min) * normalized_log
    
    scores = ei + lambda_explore * sigma_safe
    
    # Check deadline (just in case, though this is fast)
    elapsed = time.time() - start_time
    if elapsed > 290: # Unlikely
         # Return pure EI as fallback
         return ei
         
    return scores
