import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 325.0
_rng = np.random.RandomState(42)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    Draws a single posterior sample at each candidate point and scores by
    the ratio of the negative sampled function value to the evaluation cost,
    stochastically balancing exploration and exploitation.
    Returns scores of shape (2000,), where higher scores indicate better candidates.
    """
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: simple cost-aware Thompson score using mean only
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        # Sample once from posterior
        sample = mu + sigma_safe * _rng.randn(2000)
        scores = -sample / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure safety in denominators
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Thompson Sampling: draw a single sample from the posterior at each candidate
    # sample(x) ~ N(mu(x), sigma(x)^2)
    # We want to minimize f(x), so higher score = lower sampled value
    # Score = -sampled_value / cost  (more negative sampled values -> higher score)
    
    # Generate the Thompson sample
    # Use a single global RNG state for reproducibility across calls within a run
    noise = _rng.randn(2000)
    sample = mu + sigma_safe * noise

    # Score: we want to pick candidates with low sampled objective value
    # and low cost. Use ratio: -sample / cost
    # This naturally balances: low sample (good) and low cost (good)
    scores = -sample / cost_safe

    # Ensure all outputs are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
