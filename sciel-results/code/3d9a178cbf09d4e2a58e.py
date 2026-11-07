# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Thompson Sampling Acquisition.
    
    Strategy:
    1. Draw a single sample from the GP posterior for each candidate:
       sample_i = mu_i + sigma_i * z_i, where z_i ~ N(0, 1)
    2. The score for each candidate is (f_best - sample_i) / cost_i.
       This represents the expected improvement (in objective units) per unit cost.
       - If sample_i < f_best, the score is positive (candidate likely improves).
       - If sample_i > f_best, the score is negative (candidate likely worse).
       - Dividing by cost penalizes expensive evaluations.
    3. This naturally balances exploration (high sigma leads to wider sample spread,
       so sometimes a high-sigma point gets a low sample) and exploitation
       (low mu + low sigma points consistently get high scores).
    4. No heuristic tuning of exploration coefficients is needed.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Predictive mean from the GP.
    sigma : array-like, shape (2000,)
        Predictive standard deviation from the GP.
    f_best : float
        Lowest observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
        return np.zeros_like(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Draw a single sample from the posterior for each candidate
    # Using a fixed seed for reproducibility within a single call is not appropriate
    # since we want randomness to drive exploration. We use np.random directly.
    z = np.random.standard_normal(len(mu))
    samples = mu + sigma * z
    
    # Compute cost-aware improvement score
    # We want to maximize the probability/amount of improvement per unit cost.
    # (f_best - sample) is the potential improvement if we evaluate this point.
    # Dividing by cost gives improvement per unit cost.
    scores = (f_best - samples) / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
