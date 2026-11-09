import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE_SECONDS = 290.0  # 300s limit, 10s safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Args:
        mu: array of shape (2000,) with GP mean predictions.
        sigma: array of shape (2000,) with GP standard deviation predictions.
        f_best: float, the lowest observed function value.
        cost: array of shape (2000,) with the cost of evaluating each point.
        
    Returns:
        array of shape (2000,) with scores. Higher scores indicate points 
        that should be evaluated next.
    """
    # Time guard: if we are running out of time, return a safe fallback
    # (e.g., random or just the first point, but must be finite)
    if time.time() - _START_TIME > _DEADLINE_SECONDS:
        return np.ones_like(mu)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or very small costs
    # Cost is 1 + mean(x_scaled), so min cost is 1.0.
    # But safe to clip just in case.
    cost_safe = np.maximum(cost, 1e-6)
    
    # LCB for minimization: mu - kappa * sigma
    # We want to MAXIMIZE the acquisition score.
    # So we maximize: -(mu - kappa * sigma) / cost
    # Which is: (-mu + kappa * sigma) / cost
    
    # Hyperparameter kappa controls exploration vs exploitation.
    # 2.0 is a common choice for 2D problems.
    kappa = 2.0
    
    # Calculate the "potential" of each point
    # Low mu is good, high sigma is good (exploration)
    potential = -mu + kappa * sigma
    
    # Normalize by cost to prefer cheaper evaluations
    scores = potential / cost_safe
    
    # Ensure all scores are finite
    # If sigma is 0, potential is -mu. If mu is inf, potential is -inf.
    # Replace non-finite values with a very low score so they are not picked
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
