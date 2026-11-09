# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = None
_DEADLINE_MARGIN = 15.0

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 300.0
    return _DEADLINE

def _time_remaining():
    return max(0.0, _get_deadline() - time.time())

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB).
    
    Strategy:
    1. Calculate LCB: mu - kappa * sigma.
       This encourages exploration where uncertainty (sigma) is high.
       Using a fixed kappa = 1.5 provides a stable balance between exploration
       and exploitation, avoiding the potential pitfalls of adaptive kappa
       which can shift the balance suboptimally as f_best changes.
    2. Normalize by cost: score = LCB / cost.
       This makes the acquisition function cost-aware, preferring evaluations
       that offer high potential improvement per unit cost.
       
    Why LCB over EI?
    - EI with cost normalization (EI/cost) can be overly conservative in expensive regions.
    - LCB/cost maintains a stronger pull towards high-uncertainty regions even if they are slightly more expensive, 
      which is crucial for escaping local minima in the early iterations of BO.
    """
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Fixed kappa for LCB
    # Based on knowledge: fixed kappa=1.5 outperformed adaptive logic (2.43 vs 1.41)
    kappa = 1.5
    
    # Calculate Lower Confidence Bound (minimization)
    # We want to minimize f(x), so we look for low mu and high sigma.
    # LCB = mu - kappa * sigma
    lcb = mu - kappa * sigma
    
    # Cost-aware normalization
    # We want to maximize the "quality" of the candidate relative to its cost.
    # Since LCB can be negative (which is good for minimization), we need to be careful.
    # However, the standard approach for cost-aware LCB in minimization is:
    # Maximize (LCB / cost) ? 
    # Wait, if LCB is negative, dividing by cost makes it less negative (worse) if cost > 1, or more negative (better) if cost < 1.
    # Actually, the goal is to find the minimum. 
    # If we use score = LCB / cost, and we maximize score:
    # A point with LCB = -10 and cost = 1 gives score -10.
    # A point with LCB = -5 and cost = 1 gives score -5.
    # Maximizing -5 over -10 is WRONG for minimization. We want the most negative LCB.
    
    # Correction:
    # In minimization, we want to MINIMIZE the predicted value.
    # LCB gives a lower bound. We want the LCB to be as low as possible.
    # If we define score = -LCB / cost, then maximizing score corresponds to minimizing LCB/cost.
    # Let's check:
    # Candidate A: LCB = -10, cost = 1. Score = 10.
    # Candidate B: LCB = -5, cost = 1. Score = 5.
    # Max score is A. This is correct.
    
    # However, if LCB is positive (unlikely in later stages, but possible early), 
    # e.g., LCB = 5, cost = 1. Score = -5.
    # Candidate C: LCB = 2, cost = 1. Score = -2.
    # Max score is C (less bad). This is also correct.
    
    # So, score = -LCB / cost = (kappa * sigma - mu) / cost.
    
    eps = 1e-9
    # Ensure cost is not zero or negative (it should be positive by definition 1 + mean(x))
    cost_safe = np.maximum(cost, eps)
    
    score = (kappa * sigma - mu) / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
