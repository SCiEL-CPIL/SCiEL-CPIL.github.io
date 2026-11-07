import numpy as np
import time

# Global state for Thompson Sampling
_ts_state = {
    'last_time': None,
    'sampled_values': None,
    'mu_hash': None,
    'sigma_hash': None
}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Thompson Sampling.
    
    Samples a single function realization f ~ N(mu, sigma) and scores
    candidates by -f(x) / cost(x). This naturally couples exploration
    (via posterior variance) with cost efficiency.
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the GP posterior in objective units.
    sigma : array-like, shape (N,)
        Standard deviation of the GP posterior in objective units.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (N,)
        Cost of evaluating each candidate point.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores (higher is better).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is safe for division
    cost_safe = np.maximum(cost, 1e-8)
    
    # Check if we need to resample (new posterior)
    # Use a simple hash of the data to detect changes
    current_hash = (hash(tuple(mu[::100])), hash(tuple(sigma[::100])), f_best)
    
    if _ts_state['sampled_values'] is None or _ts_state['mu_hash'] != current_hash:
        # Sample from the posterior: f(x) ~ N(mu(x), sigma(x))
        # For Thompson Sampling in minimization, we want to find low values
        # We sample f and then score by -f/cost
        np.random.seed()  # Use global random state
        eps = np.random.randn(len(mu))
        sampled_f = mu + sigma * eps
        _ts_state['sampled_values'] = sampled_f
        _ts_state['mu_hash'] = current_hash
    
    sampled_f = _ts_state['sampled_values']
    
    # Score: -f(x) / cost(x)
    # Higher score means lower f(x) relative to cost
    score = -sampled_f / cost_safe
    
    # Handle non-finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
