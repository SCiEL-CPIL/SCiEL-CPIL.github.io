import time
import numpy as np

# Global deadline guard
_START_TIME = time.time()
# The evaluator kills the program after 330s. We leave a 10s margin.
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    # Fallback: Exploit best mean, adjusted for cost
    if time.time() > _DEADLINE:
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        fallback = -np.asarray(mu, dtype=np.float64) / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Ensure inputs are numpy arrays of float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero or extreme values in sampling
    # 1e-8 is a safe lower bound for standard deviation in this context
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Clamp cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: draw a sample from the posterior for each candidate
    # sample_k = mu_k + sigma_k * z_k, where z_k ~ N(0, 1)
    z = np.random.standard_normal(mu.shape)
    samples = mu + sigma_safe * z
    
    # For minimization, we want to find low values.
    # We define the score as the "improvement" over the current best,
    # adjusted by the cost of evaluation.
    # improvement = max(0, f_best - sample)
    # score = improvement / cost
    # This encourages selecting points that are likely to be better than f_best
    # and penalizes high-cost evaluations.
    
    improvement = np.maximum(0.0, f_best - samples)
    scores = improvement / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
