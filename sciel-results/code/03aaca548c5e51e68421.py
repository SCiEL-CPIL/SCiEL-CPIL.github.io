import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    This function computes the Expected Improvement for each candidate point,
    approximating the Knowledge Gradient for a single candidate step.
    The score is the EI divided by the evaluation cost, promoting cost-effective
    exploration and exploitation.
    
    EI(x) = E[max(f_best - f(x), 0)]
    For f(x) ~ N(mu, sigma^2):
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    """
    t_start = time.time()
    deadline = t_start + 1.0  # 1 second safety margin
    
    # Convert to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Ensure sigma is positive and finite
    # Use a small epsilon to avoid division by zero and numerical instability
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    
    # Ensure cost is positive and finite
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # Clip z to prevent overflow in norm.cdf/pdf if sigma is extremely small
    # However, norm.cdf/pdf are generally stable. We clip to a reasonable range.
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    
    # Calculate Phi(z) and phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical errors can cause tiny negative values)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # Using safe division
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei / cost_safe
    
    # Deadline check
    if time.time() > deadline:
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure finite output
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
