"""Cost-aware acquisition using LCB with adaptive beta for Griewank 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB) with
    adaptive exploration parameter for Griewank 2D Bayesian Optimization.
    
    LCB uses a linear combination of the mean and standard deviation,
    which is structurally different from EI or PoI. The beta parameter
    is adapted based on the relative uncertainty to balance exploration
    and exploitation.
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the GP posterior.
    sigma : array-like, shape (N,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (N,)
        Cost of evaluating each candidate point.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores (higher is better).
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive and finite
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Check for time budget - if we've used too much time, return simple scores
    elapsed = time.time() - start_time
    if elapsed > 250:
        # Fallback: simple LCB with fixed beta
        beta = 1.0
        lcb_score = beta * sigma_safe - mu
        score = lcb_score / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Adaptive beta: adjust exploration based on the relative uncertainty
    # Compute the scale of uncertainty relative to the current best
    mu_scale = np.max(np.abs(mu)) if np.max(np.abs(mu)) > 1e-10 else 1.0
    sigma_scale = np.max(sigma_safe) if np.max(sigma_safe) > 1e-10 else 1.0
    
    # Relative uncertainty: how much the std dev varies compared to the mean
    rel_uncertainty = sigma_scale / (mu_scale + 1e-10)
    
    # Beta between 0.5 and 2.0, higher when uncertainty is high
    # This adapts exploration: more exploration when the model is uncertain
    beta = 0.5 + 1.5 * min(rel_uncertainty, 1.0)
    
    # Lower Confidence Bound (LCB) for minimization
    # We maximize (beta * sigma - mu) which is equivalent to minimizing (mu - beta * sigma)
    lcb_score = beta * sigma_safe - mu
    
    # Cost-aware normalization: divide by cost to penalize expensive evaluations
    score = lcb_score / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
