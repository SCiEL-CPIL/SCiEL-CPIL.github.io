"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s hard kill

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
    
    # 2. Upper Confidence Bound (UCB)
    # UCB balances the mean prediction and the uncertainty.
    # For minimization, we want low mu and high sigma (uncertainty).
    # Standard UCB for minimization: -mu + beta * sigma
    # We increase the UCB weight slightly from 0.5 to 0.8 to encourage
    # more exploration in multimodal landscapes (Ackley, Rastrigin).
    beta = 0.8
    ucb = -mu + beta * sigma
    
    # 3. Combine signals
    # We weight EI heavily as it is directly tied to improvement probability and magnitude.
    # UCB helps in regions where mu is not yet low but sigma is high, encouraging exploration.
    # The previous weight was 0.5. We test 0.8 to see if stronger exploration helps.
    
    score_raw = ei + 0.8 * ucb
    
    # 4. Cost-aware normalization
    # Higher score means "better to evaluate".
    # We divide by cost to penalize expensive evaluations.
    score = score_raw / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
