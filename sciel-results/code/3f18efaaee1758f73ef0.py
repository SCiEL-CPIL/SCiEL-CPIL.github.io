"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on Bayesian Regret Minimization.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores. Higher scores indicate better candidates.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-8)
    
    # Bayesian Regret Minimization Proxy
    # We approximate the expected reduction in regret.
    # Regret is f_best - min(f(x), f_best).
    # We want to maximize the expected reduction.
    # The prompt suggests: E[min(f(x), f_best)] ≈ f_best - 0.5 * max(0, f_best - mu - 0.5*sigma)
    # Thus, the reduction in regret (improvement) is approximated by:
    # 0.5 * max(0, f_best - mu - 0.5*sigma)
    
    # Calculate the improvement proxy
    # This term is positive only if f_best - mu > 0.5 * sigma
    # i.e., if the mean is sufficiently lower than f_best relative to uncertainty.
    improvement_proxy = 0.5 * np.maximum(0, f_best - mu - 0.5 * sigma)
    
    # Normalize by cost
    scores = improvement_proxy / cost_safe
    
    # Handle any non-finite values (should not happen with valid inputs, but good practice)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
