import numpy as np
import time
from scipy.special import erfc, log_ndtr, ndtr

# Global state for Thompson Sampling
_TS_SEED = 42
_TS_RNG = np.random.default_rng(_TS_SEED)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function for Bayesian optimization.
    
    Samples from the posterior predictive distribution (Normal(mu, sigma)) and
    scores candidates by (f_best - sample) / cost. This naturally balances
    exploration and exploitation through the stochastic sampling.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero and numerical instability
    sigma_safe = np.clip(sigma, 1e-10, None)
    
    # Clamp cost to avoid division by zero
    cost_safe = np.clip(cost, 1e-10, None)
    
    # Thompson Sampling: sample from the posterior predictive distribution
    # Each sample is a draw from Normal(mu_i, sigma_i) for candidate i
    samples = _TS_RNG.normal(loc=mu, scale=sigma_safe)
    
    # Score: higher is better. We want to minimize, so lower samples are better.
    # The "improvement" is f_best - sample (positive when sample < f_best)
    improvement = f_best - samples
    
    # Cost-normalized score
    scores = improvement / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if this took too long, return a simpler heuristic
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        # Fallback: deterministic LCB-based score
        beta = 1.5
        lcb = mu - beta * sigma_safe
        improvement_det = np.maximum(f_best - lcb, 0.0)
        simple_scores = improvement_det / cost_safe
        simple_scores = np.where(np.isfinite(simple_scores), simple_scores, 0.0)
        return simple_scores
    
    return scores
