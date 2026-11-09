# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining normalized EI/cost with
    sigma-based exploration bonus.
    
    Parameters
    ----------
    mu     : (N,) array of GP mean predictions
    sigma  : (N,) array of GP standard deviations
    f_best : float, the best (lowest) observed value so far
    cost   : (N,) array of evaluation costs for each candidate

    Returns
    -------
    scores : (N,) array of finite scores, higher = evaluate next
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.random.rand(len(mu))
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.size
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    f_best = float(f_best)
    
    # Calculate normalized EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative (EI should be >= 0 by definition, but numerical issues)
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjusted EI: prioritize high improvement per unit cost
    # This is the core exploitation signal
    score_ei_cost = ei / cost_safe
    
    # Exploration bonus: reward uncertainty per unit cost
    # Small coefficient to not overwhelm exploitation
    beta_exploration = 0.15
    score_exploration = beta_exploration * (sigma_safe / cost_safe)
    
    # Additional small penalty for very high cost candidates to encourage
    # efficient use of budget (moderate cost penalty)
    # This helps in the small budget regime
    cost_penalty = 0.05 * (cost_safe - 1.0) / np.max(cost_safe)
    
    # Total score
    scores = score_ei_cost + score_exploration - cost_penalty
    
    # Ensure all scores are finite
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
# EVOLVE-BLOCK-END
