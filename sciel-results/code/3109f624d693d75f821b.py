import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement (Minimization)
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PoI)
    poi = cdf_z
    
    # Adaptive blending based on relative uncertainty
    # Use mean-based normalization for stability against outliers
    sigma_mean = np.mean(sigma)
    if sigma_mean > 1e-9:
        # Scale sigma by 2*mean to map typical values to ~0.5
        sigma_rel = sigma / (sigma_mean * 2.0)
        sigma_rel = np.clip(sigma_rel, 0.0, 1.0)
    else:
        sigma_rel = np.zeros_like(sigma)
        
    # Alpha controls the blend: alpha * EI + (1-alpha) * PoI
    # High uncertainty (sigma_rel ~ 1) -> Low alpha (more PoI/exploration)
    # Low uncertainty (sigma_rel ~ 0) -> High alpha (more EI/exploitation)
    # Range [0.3, 1.0] allows for stronger exploration than previous [0.5, 1.0]
    alpha = 0.3 + 0.7 * (1.0 - sigma_rel)
    
    # Combined score
    combined = alpha * ei + (1.0 - alpha) * poi
    
    # Cost-aware normalization
    # Softer penalty (exponent 1.0) to allow exploration of moderately expensive regions
    cost_safe = np.maximum(cost, 1e-9)
    scores = combined / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 325:
        return np.zeros_like(mu)
        
    return scores
