# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function combining POI and EI for Hartmann 6D."""
    deadline = time.time() + 280  # Safety margin before 300s limit
    start = time.time()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Floor sigma to avoid division by zero and excessive exploration
    sigma = np.maximum(sigma, 1e-5)
    
    # Standardize for numerical stability
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z for numerical stability
    z_clipped = np.clip(z, -10, 10)
    
    # POI: probability of improvement
    poi = norm.cdf(z_clipped)
    
    # EI: expected improvement
    ei = improvement * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Cost adjustment
    inv_cost = 1.0 / cost
    
    # Hybrid: blend POI and EI, weighted toward POI for conservatism
    # POI is in [0,1], EI can be larger; normalize EI contribution
    # Use a weighted combination that favors POI (conservative) but retains some EI signal
    poi_score = poi * inv_cost
    ei_score = ei * inv_cost
    
    # Blend: 70% POI, 30% EI (tuned for conservative behavior in 6D multimodal)
    score = 0.7 * poi_score + 0.3 * ei_score
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard: if we've exceeded our safety deadline, return what we have
    if time.time() > deadline:
        return score
    
    return score
# EVOLVE-BLOCK-END
