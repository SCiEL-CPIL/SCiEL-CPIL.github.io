"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function for minimization.
    
    Strategy:
    1. Compute the Expected Improvement (EI) for each candidate point.
       For minimization, EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
       where z = (f_best - mu(x)) / sigma(x).
       Phi is the standard normal CDF, and phi is the standard normal PDF.
       This measures the expected amount by which the function value at x
       is expected to be lower than the current best f_best.
       
    2. Make the acquisition function cost-aware by dividing the EI by the
       evaluation cost. Since cost is 1 + mean(x_scaled), points closer to
       the origin (lower cost) are favored when EI is comparable.
       
    3. Return the cost-adjusted EI scores. Higher scores indicate better
       candidates to evaluate next.
    """
    # Check for finite inputs
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and 
            np.all(np.isfinite(f_best)) and np.all(np.isfinite(cost))):
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative and handle tiny values to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute z-score: how many standard deviations f_best is above mu
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute EI for minimization:
    # EI = (f_best - mu) * P(y < f_best) + sigma * E[y | y < f_best] - sigma * P(y < f_best) ... wait
    # Standard EI formula for minimization:
    # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # Phi(z) is the probability that y < f_best
    # phi(z) is the standard normal PDF at z
    
    # We use scipy.stats.norm for accuracy
    phi_z = norm.pdf(z)
    phi_z = np.nan_to_num(phi_z, nan=0.0, posinf=0.0, neginf=0.0)
    
    # For very large positive z, Phi(z) -> 1, EI -> (f_best - mu)
    # For very large negative z, Phi(z) -> 0, EI -> 0
    # We can compute Phi(z) directly.
    phi_cdf = norm.cdf(z)
    phi_cdf = np.nan_to_num(phi_cdf, nan=0.0, posinf=1.0, neginf=0.0)
    
    ei = (f_best - mu) * phi_cdf + sigma_safe * phi_z
    
    # Ensure EI is non-negative (it should be by definition, but numerical errors might occur)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    scores = ei / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
