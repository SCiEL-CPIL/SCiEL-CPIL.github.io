"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    
    EI for minimization:
    EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) 
             + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    
    where Phi is the CDF and phi is the PDF of the standard normal distribution.
    
    We return the score: EI(x) / cost(x).
    Higher scores indicate points that are expected to improve the best observed value
    the most per unit of cost.
    
    We include a deadline guard to ensure we return within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin of 10s before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Initialize scores to 0
    scores = np.zeros(n, dtype=np.float64)
    
    # Safety check for finite inputs
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return scores
    
    # Ensure cost is positive to avoid division by zero or negative costs
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate the z-score for EI
    # z = (f_best - mu) / sigma
    # Handle sigma = 0 case
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate Phi(z) and phi(z)
    # Using scipy.stats.norm for accuracy
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI formula:
    # If f_best - mu > 0 (potential improvement), EI > 0
    # If f_best - mu <= 0, EI should be 0 (no expected improvement)
    # The standard formula handles this, but we can explicitly zero out negative improvements
    # to avoid numerical noise, although mathematically EI >= 0 always.
    
    improvement = f_best - mu
    ei = improvement * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Score is EI per cost
    scores = ei / cost_safe
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.time() > deadline:
        # If we somehow took too long, return what we have (which is already computed)
        pass
        
    return scores
# EVOLVE-BLOCK-END
