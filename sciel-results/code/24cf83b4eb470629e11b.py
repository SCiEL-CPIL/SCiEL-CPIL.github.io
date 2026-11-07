import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition.
    Samples a single function realization from the GP posterior and scores
    candidates by -f_sample / cost, naturally coupling exploration with cost efficiency.
    
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
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Thompson Sampling: sample one function realization from the GP posterior
    # f_sample(x) ~ N(mu(x), sigma(x)^2)
    rng = np.random.default_rng()
    noise = rng.standard_normal(len(mu))
    f_sample = mu + sigma_safe * noise
    
    # Score by negative predicted value divided by cost
    # Higher score = more desirable (lower predicted value, lower cost)
    score = -f_sample / cost_safe
    
    # Handle non-finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
