"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_START_TIME = None
_DEADLINE = 330.0  # Wall clock limit
_SAFETY_MARGIN = 5.0  # Seconds before limit to stop

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME, _DEADLINE, _SAFETY_MARGIN
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    _init_time()
    
    # Ensure inputs are numpy arrays for speed
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check: if we are running out of time, return a deterministic fallback
    if _time_exceeded():
        # Fallback: return a simple EI score
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    n = len(mu)
    
    # Strategy: Cost-Aware Lower Confidence Bound (LCB)
    #
    # The current program uses Probability of Improvement (PI) with a quadratic cost penalty.
    # PI is a binary-like metric (probability) that ignores the magnitude of potential improvement.
    # In rugged landscapes like Rastrigin or Ackley, the magnitude of improvement can be crucial,
    # especially when f_best is far from the global minimum.
    #
    # LCB (Lower Confidence Bound) for minimization is defined as:
    #   LCB(x) = mu(x) - kappa * sigma(x)
    # We want to minimize LCB(x), so the acquisition score (to be maximized) is:
    #   Score = -LCB(x) / Cost(x) = (kappa * sigma(x) - mu(x)) / Cost(x)
    #
    # This is structurally different from PI because:
    # 1. It is unbounded (unlike PI which is in [0,1]).
    # 2. It scales linearly with both the predicted mean and the uncertainty.
    # 3. It directly optimizes the trade-off between exploration (sigma) and exploitation (mu).
    # 4. The cost normalization is linear (1/cost) rather than quadratic (1/cost^2),
    #    which is a different structural choice in how cost is penalized.
    #
    # Choice of kappa:
    # A common heuristic is to set kappa based on the problem scale or to use a fixed value.
    # Since we don't have access to the number of iterations or the specific problem scale
    # (other than f_best and mu/sigma distributions), we can use a fixed kappa that balances
    # exploration and exploitation.
    #
    # Let's try kappa = 1.0. This is a moderate exploration level.
    # If kappa is too high, we explore too much and ignore good means.
    # If kappa is too low, we exploit too much and get stuck in local optima.
    #
    # Another consideration: the scale of mu and sigma.
    # If mu and sigma are very small, kappa=1 might be too large.
    # However, the GP is standardized in the task description ("standardized targets"),
    # so mu and sigma should be on a comparable scale to the objective function's scale.
    #
    # Let's implement LCB with kappa = 1.0 and linear cost normalization.
    
    cost_safe = np.maximum(cost, 1.0)
    
    # Compute LCB score
    kappa = 1.0
    lcb_score = (kappa * sigma - mu) / cost_safe
    
    # Ensure finite scores
    lcb_score = np.nan_to_num(lcb_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return lcb_score
# EVOLVE-BLOCK-END
