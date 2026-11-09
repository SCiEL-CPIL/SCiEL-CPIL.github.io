import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)
    
    # z-score for minimization
    z = (f_best - mu) / sigma_safe
    
    # CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PoI)
    poi = cdf_z
    poi = np.clip(poi, 0.0, 1.0)
    
    # Cost-awareness: normalize EI by cost to prefer cheaper evaluations
    # Also add a small exploration term to avoid getting stuck
    sigma_median = np.median(sigma_safe)
    exploration = sigma_median * 0.1  # small exploration bonus
    
    # Combine: cost-normalized EI with exploration
    # The idea is that we want to balance improvement magnitude, confidence, and cost
    # score = (ei + exploration) / cost
    # This makes cheaper points more attractive while still favoring high EI
    
    score = (ei + exploration) / cost_safe
    
    # Additional boost for high-confidence improvements
    # Multiply by (1 + 0.5 * poi) to slightly favor confident predictions
    score = score * (1.0 + 0.5 * poi)
    
    # Ensure finite outputs
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 290:
        return np.zeros_like(mu)
        
    return score
