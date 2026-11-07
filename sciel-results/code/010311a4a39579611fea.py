"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE = 330.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Strategy:
    1. Compute Expected Improvement (EI) for minimization:
       EI(x) = E[max(f_best - f(x), 0)]
       For a Gaussian posterior N(mu, sigma^2):
       EI = (f_best - mu) * Phi(d) + sigma * phi(d)
       where d = (f_best - mu) / sigma
       Phi is the standard normal CDF, phi is the standard normal PDF.
       
    2. Cost Awareness:
       The cost of evaluating a point is given by `cost`.
       We want to maximize the "value" per unit cost.
       A common heuristic is to divide the acquisition value by the cost.
       However, simple division can be unstable if cost varies widely or if EI is small.
       Another approach is to use a weighted combination or a penalty.
       
       Given the task description "Cost-aware acquisition", and that we want to
       minimize the objective, a robust approach is:
       Score = EI / cost
       
       Why? If a point has high EI but very high cost, the score is reduced.
       If a point has moderate EI but very low cost, the score is boosted.
       This aligns with the "cost-aware" requirement.
       
       To ensure stability and encourage exploration when EI is low (early on or in uncertain regions),
       we can add a small exploration term based on sigma, also cost-adjusted.
       However, pure EI already contains an exploration component via the sigma * phi(d) term.
       
       Let's stick to Score = EI / cost for clarity and theoretical grounding.
       If cost is 0, we should handle it, but cost is 1 + mean(x_scaled), so min cost is 1.0.
       
    3. Safety:
       - Ensure sigma is positive (clip to 1e-8).
       - Ensure finite outputs.
       - Check time limit.
    """
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
         # Return a neutral score to avoid hanging or invalid computation
         return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero or numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute d = (f_best - mu) / sigma
    # Note: f_best is the minimum observed so far.
    # We are minimizing, so improvement is f_best - f(x).
    # If f(x) < f_best, improvement is positive.
    d = (f_best - mu) / sigma
    
    # Compute EI
    # EI = (f_best - mu) * Phi(d) + sigma * phi(d)
    # Phi(d) = norm.cdf(d)
    # phi(d) = norm.pdf(d)
    
    # Optimization: Use stable computation if possible, but for 2000 points it's fine.
    # norm.cdf and norm.pdf are vectorized in scipy.
    phi_d = norm.pdf(d)
    Phi_d = norm.cdf(d)
    
    EI = (f_best - mu) * Phi_d + sigma * phi_d
    
    # Ensure EI is non-negative (numerical errors might make it slightly negative)
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware score
    # We want to maximize this score.
    # Dividing by cost penalizes expensive points.
    cost_safe = np.maximum(cost, 1e-9)
    scores = EI / cost_safe
    
    # Optional: Add a small exploration bonus to avoid getting stuck in local optima
    # The EI term already explores via sigma, but in regions where f_best is far from mu,
    # EI might be dominated by the mean.
    # A pure EI is generally robust. Let's stick to it for now.
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
