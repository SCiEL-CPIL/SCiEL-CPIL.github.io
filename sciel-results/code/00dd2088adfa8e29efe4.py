import numpy as np
import time
from scipy.special import erf

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1.0)  # cost is in [1, 2]
    
    # Check deadline
    if time.time() > deadline:
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -50, 50)
        phi_z = np.exp(-0.5 * z_clipped ** 2) / np.sqrt(2 * np.pi)
        cdf_z = 0.5 * (1.0 + erf(z_clipped / np.sqrt(2.0)))
        ei = (f_best - mu) * cdf_z + sigma * phi_z
        ei = np.maximum(ei, 0.0)
        scores = ei / np.log(1.0 + cost_safe)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -50, 50)
    
    # Compute phi(z) and Phi(z)
    phi_z = np.exp(-0.5 * z_clipped ** 2) / np.sqrt(2 * np.pi)
    cdf_z = 0.5 * (1.0 + erf(z_clipped / np.sqrt(2.0)))
    
    # Expected Improvement
    ei = (f_best - mu) * cdf_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        scores = ei / np.log(1.0 + cost_safe)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Cost-aware Expected Improvement (CEI)
    # Use linear cost as the penalty
    cost_penalty = cost_safe
    
    # Base CEI score
    # Use a simple additive exploration term (UCB-like)
    # kappa controls the balance between exploitation (EI) and exploration (sigma)
    kappa = 0.1
    exploration_term = kappa * sigma
    
    scores = (ei + exploration_term) / cost_penalty
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    if time.time() > deadline:
        return np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
