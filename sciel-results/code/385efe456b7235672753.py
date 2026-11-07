"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time to track total elapsed time for the whole run
_START_TIME = time.time()
TIME_LIMIT = 330.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    
    Structural Change:
    Replaces Expected Improvement (EI) with Cost-Effective Upper Confidence Bound (CE-UCB).
    EI is greedy and can get stuck in local optima or ignore high-variance regions that might contain the global minimum.
    UCB balances exploitation (low mu) and exploration (high sigma) explicitly.
    The cost is incorporated by dividing the UCB score by the cost raised to a power alpha,
    effectively maximizing the expected improvement per unit cost.
    
    The exploration parameter kappa is dynamic, decreasing as f_best improves (progress),
    shifting from aggressive exploration to exploitation.
    """
    # Deadline Guard
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT - 5.0:
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        # Fallback: simple cost-aware mean
        return np.maximum(-mu / cost, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Dynamic Kappa based on f_best
    # f_best starts high (e.g., 50) and goes to 0.
    # We want high kappa (exploration) when f_best is high.
    # We want low kappa (exploitation) when f_best is low.
    
    f_best_clamped = np.clip(f_best, -100, 100)
    
    # Sigmoid mapping
    # arg = a * (f_best - b)
    # If f_best > b, arg > 0, sigmoid ~ 1.
    # If f_best < b, arg < 0, sigmoid ~ 0.
    a_param = 0.2
    b_param = 5.0
    
    arg = a_param * (f_best_clamped - b_param)
    # Avoid overflow in exp
    arg_clipped = np.clip(arg, -50, 50)
    sigmoid_val = 1.0 / (1.0 + np.exp(-arg_clipped))
    
    kappa_min = 0.5
    kappa_max = 2.5
    kappa = kappa_min + (kappa_max - kappa_min) * sigmoid_val
    
    # UCB for minimization: -mu + kappa * sigma
    # We want to maximize this score.
    base_score = -mu + kappa * sigma_safe
    
    # Cost penalty
    # Cost is in [1, 2].
    # Dividing by cost^alpha penalizes expensive points.
    alpha = 1.2
    score = base_score / (cost ** alpha)
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
