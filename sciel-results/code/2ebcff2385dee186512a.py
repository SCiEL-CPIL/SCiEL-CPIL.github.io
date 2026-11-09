import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track elapsed time and ensure we return well before the limit.
_DEADLINE = 325.0  # seconds, safety margin of 5s
_start_time = time.time()


def _remaining_time():
    return _DEADLINE - (time.time() - _start_time)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PI) acquisition function.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate

    Returns:
        scores: array of shape (2000,) - acquisition values (higher = better)
    """
    # Safety check
    if _remaining_time() < 0.5:
        return np.zeros_like(mu)

    n = len(mu)
    
    # Ensure sigma is positive to avoid division by zero or log issues
    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Probability of Improvement (PI) for minimization:
    # We want P(f(x) < f_best).
    # Assuming f(x) ~ N(mu, sigma^2), 
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # where Phi is the CDF of the standard normal distribution.
    
    z = (f_best - mu) / sigma
    
    # Compute the CDF of the standard normal distribution
    # Using scipy.stats.norm.cdf for accuracy
    prob_improvement = norm.cdf(z)
    
    # Cost-aware adjustment:
    # The task asks for a "cost-aware" acquisition. 
    # A common approach is to divide by the cost, or multiply by 1/cost.
    # Since higher score = better, and we want to penalize high cost, 
    # we multiply the probability by (1 / cost).
    # Alternatively, we could use a cost-aware PI: 
    # score = prob_improvement / cost
    # This effectively says: "What is the probability of improvement per unit cost?"
    
    scores = prob_improvement / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
