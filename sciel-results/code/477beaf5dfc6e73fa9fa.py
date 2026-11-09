"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a cost-normalized Lower Confidence Bound (LCB) strategy.
    The LCB balances exploration (via the uncertainty term) and exploitation 
    (via the mean) by minimizing the lower bound of the function value.
    Dividing by cost prioritizes candidates that offer high potential improvement 
    per unit of evaluation cost.
    """
    # Time guard: track elapsed time
    t_start = time.time()
    
    # Ensure sigma is positive to avoid division/sqrt issues
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # Compute Lower Confidence Bound
    # LCB = mu - beta * sigma
    # Since we want to maximize the acquisition score, and we are minimizing f,
    # we want to maximize (f_best - LCB) or simply minimize LCB.
    # However, standard LCB acquisition is often defined as maximizing -LCB 
    # or maximizing (f_best - LCB).
    # Let's use the standard form for minimization: maximize (f_best - (mu - beta*sigma)).
    # This is equivalent to maximizing (f_best - mu + beta*sigma).
    # Alternatively, we can just use the negative of the LCB value if we assume 
    # the scale is comparable, but f_best - mu is more robust to shifts.
    # Let's stick to the "improvement" style but with LCB structure.
    # Actually, the prompt suggests: "directly optimizes the posterior lower bound 
    # (mu - beta * sqrt(var)) divided by the evaluation cost".
    # Wait, if we minimize f, the "lower bound" is the optimistic guess of how low f could be.
    # So we want to pick the x that minimizes mu - beta*sigma.
    # To turn this into a maximization problem (higher score = better), we can use:
    # Score = - (mu - beta * sigma) / cost
    # Or, to incorporate f_best for scale invariance similar to EI:
    # Score = (f_best - (mu - beta * sigma)) / cost
    
    # Let's use the latter as it is more stable and comparable to EI/POI.
    # Beta controls the exploration-exploitation tradeoff.
    # A fixed beta might be suboptimal. A common heuristic is beta = log(t) or similar,
    # but here we don't have 't' directly, only cost.
    # Let's use a moderate beta. For Hartmann 6D, 1.0-2.0 is typical.
    # Let's try beta = 1.5.
    beta = 1.5
    
    lcb = mu - beta * sigma
    
    # We want to maximize the potential improvement over f_best, normalized by cost.
    # Improvement = f_best - lcb
    # If lcb < f_best, improvement is positive.
    # If lcb > f_best, improvement is negative (we shouldn't pick this).
    # The division by cost encourages picking cheap points.
    
    score = (f_best - lcb) / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time check: if we're taking too long, return early with current result
    t_elapsed = time.time() - t_start
    if t_elapsed > 5.0:  # Safety margin
        return score
    
    return score
# EVOLVE-BLOCK-END
