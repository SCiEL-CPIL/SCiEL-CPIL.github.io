import numpy as np
from scipy.stats import norm
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE_MARGIN = 5.0  # seconds to leave before the 330s kill
_START_TIME = None


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Thompson Sampling with cost-awareness: sample from the posterior
    and select candidates with low sampled values, adjusted for cost.
    This is a fundamentally different exploration-exploitation paradigm
    that can escape local minima more effectively in high dimensions.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1])).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = prefer to evaluate next).
    """
    # Deadline guard: if we're getting close to the limit, return a safe fallback
    if _elapsed() > 330 - _DEADLINE_MARGIN:
        # Fallback: simple LCB/cost
        sigma_safe = np.maximum(sigma, 1e-9)
        lcb = mu - 1.0 * sigma_safe
        scores = -lcb / (cost + 1e-9)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure inputs are finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-8, None)
    
    # Thompson Sampling: draw a single sample from the posterior
    # sample = mu + sigma * z, where z ~ N(0,1)
    # We use a fixed random seed for reproducibility within a call
    rng = np.random.default_rng(seed=42)
    z = rng.standard_normal(len(mu))
    sample = mu + sigma * z
    
    # For minimization, we want low sampled values
    # Score = -sample (higher is better for selection)
    # Cost-aware: divide by cost to prefer cheaper evaluations
    cost_safe = np.clip(cost, 1e-6, None)
    scores = -sample / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
