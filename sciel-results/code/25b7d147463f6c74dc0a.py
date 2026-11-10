import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_TIME_LIMIT = 300.0  # Safety margin before 330s hard kill
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and Upper Confidence Bound (UCB) signals, normalized by evaluation cost.
    
    This approach balances exploitation (finding points likely to improve
    on the current best) with exploration (visiting uncertain regions)
    while respecting the non-uniform cost structure of the problem.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far (lower is better).
    cost : np.ndarray, shape (2000,)
        Evaluation cost for each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = better to evaluate next).
    """
    global _call_count
    _call_count += 1
    
    # Time guard: if we're running low on time, return a simple fallback
    if time.time() - _start_time > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        # Fallback: Pure EI / Cost
        fallback = ei / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # 1. Expected Improvement (EI)
    # EI is the expected improvement over f_best.
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Probability of Improvement (PI)
    # PI is the probability that the point beats f_best.
    pi = norm.cdf(z)
    
    # 3. Cost-Discounted PI (CD-PI)
    # Normalize by sqrt(cost) to balance exploration frequency vs. cost.
    cd_pi = pi / np.sqrt(cost)
    
    # 4. Upper Confidence Bound (UCB)
    # For minimization: -mu + beta * sigma
    # Use a small beta to allow some exploration of uncertain regions.
    beta = 0.3
    ucb = (-mu + beta * sigma) / cost
    
    # 5. Combine signals
    # Blend EI/cost (magnitude of improvement) with CD-PI (frequency of improvement)
    # and a small UCB term for exploration.
    # Weight EI/cost heavily as it captures both probability and magnitude.
    # CD-PI provides a robust signal for frequent small improvements.
    
    ei_cost = ei / cost
    
    # Combine: primarily EI/cost, with some CD-PI and UCB
    score = 0.7 * ei_cost + 0.2 * cd_pi + 0.1 * ucb
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
