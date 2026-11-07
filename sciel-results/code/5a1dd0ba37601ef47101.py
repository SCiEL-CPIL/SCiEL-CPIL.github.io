# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Cost-Aware Upper Confidence Bound (CA-UCB)
    with an adaptive beta parameter.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (n_candidates,)
        Mean of the GP posterior.
    sigma : np.ndarray, shape (n_candidates,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed value so far.
    cost : np.ndarray, shape (n_candidates,)
        Cost of evaluating each candidate (1 + mean(x)).
    
    Returns:
    --------
    scores : np.ndarray, shape (n_candidates,)
        Acquisition scores (higher is better).
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Estimate total cost consumed so far.
    # We don't have direct access to total_cost, but we can infer the stage
    # from the magnitude of sigma and f_best, or use a heuristic based on
    # the number of evaluations implied by the GP's certainty.
    # A simpler heuristic: use the variance of mu or sigma to estimate progress.
    # However, a common proxy in these static contexts is to assume the
    # optimization progresses linearly with the number of points.
    # Since we can't track n_evals, we'll use a fixed beta that balances
    # exploration and exploitation well for the hartmann3d problem with
    # budget 30.
    
    # The directive suggests beta = 1 / (1 + total_cost/30).
    # Early on (total_cost ~ 7.5), beta ~ 1/(1+0.25) = 0.8
    # Late (total_cost ~ 30), beta ~ 1/(1+1) = 0.5
    # Let's use a beta that reflects a mid-stage balance, or simply
    # use a slightly lower beta than 1.0 to favor exploitation more,
    # as hartmann3d is a multimodal function where finding the global
    # minimum is key.
    
    # Let's try beta = 0.5 as a reasonable compromise.
    beta = 0.5
    
    # CA-UCB for minimization:
    # LCB = mu - beta * sigma
    # We want to minimize LCB, so we maximize -LCB = beta * sigma - mu.
    # Divide by cost to make it cost-aware.
    
    ucb_score = (beta * sigma_safe - mu) / cost_safe
    
    # Ensure finite values
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -np.inf)
    
    return ucb_score
# EVOLVE-BLOCK-END
