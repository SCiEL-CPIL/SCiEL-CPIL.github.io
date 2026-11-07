"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 325.0  # 5s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining Expected Improvement (EI) with cost normalization.
    
    Uses the structural idea of Cost-Effective EI (EI / cost) from inspiration 3,
    combined with the fast erf-based computation and deadline guard from the current program.
    
    The key insight: dividing EI by cost directly optimizes expected improvement per unit
    budget spent, which is the right objective for cost-aware BO.
    """
    # Deadline guard: if we're running out of time, return simple exploitation
    if time.time() > _DEADLINE:
        safe_cost = np.maximum(cost, 1e-6)
        scores = -mu / safe_cost
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle NaN/Inf
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -50, 50)
    
    # Fast CDF and PDF using erf
    sqrt_2 = np.sqrt(2.0)
    cdf = 0.5 * (1.0 + erf(z / sqrt_2))
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    # Expected Improvement: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Ensure EI is non-negative (it should be by construction, but guard)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Effective EI: maximize EI per unit cost
    # This directly optimizes expected improvement relative to budget spent
    scores = ei / cost
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
