"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track total elapsed time for the whole run
_START_TIME = time.time()
TIME_LIMIT = 330.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    
    Structural Change:
    Combines the Expected Improvement (EI) formulation from the high-scoring inspiration
    with the dynamic confidence-based exploration parameter from the UCB approach.
    
    1. Computes Expected Improvement (EI) for minimization.
    2. Normalizes EI by the candidate's cost to get a "value-for-money" metric (EIPC).
    3. Applies a dynamic penalty for uncertainty (sigma) weighted by a confidence factor
       derived from f_best. Low f_best -> High Confidence -> High penalty for uncertainty.
       High f_best -> Low Confidence -> Low penalty (more exploration).
    
    This shifts from a pure UCB or pure EI to a cost-normalized EI with adaptive risk aversion.
    """
    # Deadline Guard: Check if we are close to the time limit.
    # If so, return a simple, fast score to allow the loop to finish gracefully.
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT - 5.0:
        # Fallback: Simple Expected Improvement, ignoring cost to save time/logic
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Expected Improvement for minimization: E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic Beta based on f_best
    # We map f_best to a "Confidence" score C in [0, 1].
    # High f_best -> Low Confidence -> Low Beta (Explore, ignore uncertainty penalty more)
    # Low f_best  -> High Confidence -> High Beta (Exploit, penalize uncertainty heavily)
    
    # Heuristic parameters for the sigmoid:
    # Assume f_best starts high (e.g., 50-100) and drops to 0.
    # We want the transition to happen when f_best is reasonably low, say < 10.
    # C = 1 / (1 + exp(a * (f_best - b)))
    # If f_best >> b, C -> 0.
    # If f_best << b, C -> 1.
    
    a_param = 0.5
    b_param = 5.0
    
    # Clamp f_best for numerical stability in exp
    f_best_clamped = np.clip(f_best, -100, 100)
    
    confidence = 1.0 / (1.0 + np.exp(a_param * (f_best_clamped - b_param)))
    
    # Beta ranges from 0.1 (low confidence) to 1.0 (high confidence)
    beta_min = 0.1
    beta_max = 1.0
    beta = beta_min + confidence * (beta_max - beta_min)
    
    # Compute the acquisition score
    # Strategy: EIPC (Expected Improvement per Cost) minus Uncertainty Penalty
    # 
    # Term 1: ei / cost_safe
    #   This normalizes the expected gain by the cost.
    #   A point with EI=1 and Cost=1 scores 1.0.
    #   A point with EI=1 and Cost=2 scores 0.5.
    #   This inherently prefers cheaper points when EI is comparable.
    #
    # Term 2: - beta * sigma_safe
    #   This penalizes uncertainty.
    #   By NOT multiplying by cost here, we avoid double-penalizing expensive
    #   uncertain points (they are already penalized in Term 1).
    #   The penalty is purely for the risk of the prediction being wrong.
    
    eipc = ei / cost_safe
    
    score = eipc - beta * sigma_safe
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
