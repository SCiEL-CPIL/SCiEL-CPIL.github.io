import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PoI)
    poi = cdf_z
    
    # Cost-aware scoring
    cost_safe = np.maximum(cost, 1e-9)
    
    # Use a softer cost penalty to encourage more exploration in early stages
    # when cost is low, but still penalize expensive points.
    # Using cost**0.5 instead of cost**1 reduces the penalty for moderate costs.
    # Also, we can add a small exploration bonus based on sigma to avoid
    # getting stuck in low-uncertainty regions that aren't necessarily optimal.
    
    # Exploration bonus: proportional to sigma, but dampened if mu is already good
    # This encourages looking at high-uncertainty areas even if EI is low.
    exploration_bonus = sigma_safe * 0.1
    
    # Combined score: (EI + Exploration Bonus) / cost**0.5
    # The exploration bonus helps escape local optima by valuing uncertainty.
    # The softer cost exponent allows more evaluations of moderately expensive points.
    
    scores = (ei + exploration_bonus) / np.sqrt(cost_safe)
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 325:
        return np.zeros_like(mu)
        
    return scores
