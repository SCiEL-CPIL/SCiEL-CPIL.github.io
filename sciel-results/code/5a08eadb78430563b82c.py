"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def acquisition(mu, sigma, f_best, cost):
    """
    Budget-aware hybrid acquisition function for Bayesian optimization on griewank 2d.
    
    Parameters
    ----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard: if we're getting close to the 330s limit, return a safe fallback
    if _elapsed() > 320:
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        # Fallback: exploit low mu, penalize high cost
        return -mu / np.maximum(cost, 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values and safe denominators
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Calculate remaining budget fraction
    # Total budget is 30. We don't know current total cost exactly, but we can estimate.
    # Actually, 'cost' here is the cost of the *candidate*, not the total spent.
    # The task says: "While total cost is below 30...". 
    # We don't have access to the *current* total cost directly in the function signature.
    # However, we can infer a proxy or just use a fixed schedule based on the number of evaluations?
    # No, we don't have the count either.
    # 
    # Let's look at the inspirations. 
    # Inspiration 1: "dynamic, budget-aware exploration coefficient (shrinking as remaining cost de..."
    # Since we don't have total_cost or n_evals, we must rely on the properties of mu/sigma or just use a fixed heuristic that works well.
    #
    # Alternative: Use the distribution of sigma to calibrate exploration.
    # In early stages, sigma is high everywhere. In later stages, sigma is low near known points.
    # 
    # A robust approach without explicit budget tracking:
    # Use a UCB-like score: -mu + beta * sigma
    # But we need to account for cost.
    # Score = (-mu + beta * sigma) / cost
    #
    # How to choose beta?
    # If we use a fixed beta, it might be too explorative or too exploitative.
    # Let's try to make beta adaptive based on the current uncertainty level relative to f_best.
    #
    # Let's try a different angle: Expected Improvement is often better for minimization.
    # But EI can be flat.
    # 
    # Let's try a "Cost-Adjusted UCB" with a beta that scales with the ratio of current uncertainty to best value.
    # However, f_best starts high (Griewank max is ~1.0 in [0,1] scaled? No, Griewank in [-10,10] has max ~1.0? 
    # Griewank 2D: f(x) = 1 - prod(cos(x_i/sqrt(i)) / sqrt(i)). 
    # In [-10,10], values are in [0, ~1]. Min is 0.
    # So f_best is in [0, 1].
    #
    # Let's use a fixed beta that is tuned for this problem.
    # The inspiration 1 scored 1.4563 with "Cost-Aware UCB".
    # Let's implement a Cost-Aware UCB.
    #
    # Score = ( -mu + beta * sigma ) / cost
    #
    # What beta?
    # If beta is too high, we explore noise. If too low, we get stuck.
    # A common choice is beta = 2 * sigma_max or similar.
    # But we have 2000 points.
    # Let's try beta = 1.0 initially.
    
    beta = 1.0
    
    # Calculate UCB score
    ucb_scores = -mu + beta * sigma
    
    # Cost normalization
    # Dividing by cost encourages cheap evaluations.
    # However, if cost is very low, this might overvalue cheap but uninformative points.
    # A softer penalty: divide by sqrt(cost) or (1 + cost)?
    # The seed program used ei / cost.
    # Let's stick to / cost for consistency with the "cost-aware" theme, but UCB might behave differently.
    
    scores = ucb_scores / cost
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
