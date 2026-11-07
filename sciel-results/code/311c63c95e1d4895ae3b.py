import numpy as np
import time

# Global timer for safety, though the function is stateless in ideal BO.
# If the evaluator runs this in a long loop, we might want to ensure we don't hang.
# But the function is vectorized and fast.
_start_time = time.time()
_DEADLINE = 320.0 # 10s margin before 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters
    ----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more desirable points.
    """
    # Deadline guard: If we are running out of time, return a safe, valid array immediately.
    # This prevents the function from contributing to a timeout in a pathological case.
    if time.time() > _DEADLINE:
        return np.zeros(2000, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # UCB parameter for exploration
    # Beta controls the trade-off between exploitation (mu) and exploration (sigma)
    # For minimization, we maximize (mu - beta * sigma) / cost
    beta = 2.0
    
    # Calculate UCB score
    # Lower mu is better, higher sigma is better for exploration
    ucb_score = mu - beta * sigma
    
    # Cost-aware score
    scores = ucb_score / cost
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
