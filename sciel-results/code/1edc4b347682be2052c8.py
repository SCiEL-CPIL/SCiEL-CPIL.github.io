# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for time tracking
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for 2000 candidates.
    Uses a Thompson Sampling variant: samples a target from N(f_best, max_sigma)
    and scores by (target - mu) / cost.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard: if we're getting close to the 330s limit, return safe defaults
    # The function is called within the evaluator loop, so we track time here.
    # If this function is called too late, we might want to bail out, but the contract
    # says "Each evaluation must finish within 300 s". The caller handles the loop.
    # However, to be safe against pathological cases, we ensure we don't hang.
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety clamps
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling Strategy:
    # 1. Determine the exploration scale based on the current maximum uncertainty.
    #    If uncertainty is high, we explore more broadly.
    max_sigma = np.max(sigma)
    
    # 2. Sample a scalar target value.
    #    We center the distribution at f_best.
    #    The standard deviation is scaled by max_sigma to adapt exploration to the landscape.
    #    A coefficient of 1.0 means we sample from N(f_best, max_sigma).
    #    This effectively shifts the "goal" for improvement.
    target = np.random.normal(loc=f_best, scale=max_sigma)
    
    # 3. Calculate the "improvement" relative to this stochastic target.
    #    We want to find points where mu is significantly lower than the target.
    #    Score = (target - mu) / cost
    #    This is a linear approximation of the expected improvement relative to the sampled target.
    #    Note: Standard TS samples a function f(x) ~ N(mu, sigma) and minimizes f(x)/cost.
    #    Here, we approximate by sampling a scalar threshold.
    #    Let's refine: If we sample a full function, we'd do f_sample = mu + sigma * z.
    #    But the directive suggests: "scoring each candidate as -(mu - offset) / cost".
    #    This is equivalent to (offset - mu) / cost.
    
    scores = (target - mu) / cost
    
    # 4. Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # 5. Optional: Add a small epsilon to avoid exact ties or numerical issues?
    #    Not strictly necessary if we just return the scores.
    
    return scores
# EVOLVE-BLOCK-END
