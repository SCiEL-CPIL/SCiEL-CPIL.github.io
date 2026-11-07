import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
TIME_LIMIT = 290.0  # Safety margin before 300s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines Thompson Sampling-style stochastic score with a small
    bounded exploration term, normalized by evaluation cost.
    
    The Thompson Sampling sample provides implicit exploration-exploitation
    balance: when sigma is large, the sample can be much lower than mu,
    pushing exploration; when sigma is small, the sample is close to mu,
    favoring exploitation of known good regions.
    """
    # Check time budget
    if time.time() - _start_time > TIME_LIMIT:
        # Fallback: simple EI divided by cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: sample one value from the Gaussian posterior
    # for each candidate. This gives a single scalar score per point
    # that naturally balances exploration (high sigma -> wide sample range)
    # and exploitation (low sigma -> sample near mu).
    # Since we want to minimize, lower sampled values are better.
    # We use a deterministic seed based on the call count to avoid
    # excessive randomness while still varying across calls.
    # For reproducibility within a single call, use a fixed random state
    # derived from the shapes (deterministic per call shape).
    rng = np.random.default_rng(42)
    ts_sample = mu + sigma_safe * rng.standard_normal(mu.shape)
    
    # The Thompson score: we prefer points where the sampled value is low.
    # So the score should be negative of the sample (higher = better).
    ts_score = -ts_sample
    
    # Add a small bounded exploration bonus to encourage visiting
    # uncertain regions, scaled so it doesn't dominate.
    # The bounded form sigma/(sigma + 0.1) saturates at 1.0 and is
    # negligible when sigma is small.
    explore = sigma_safe / (sigma_safe + 0.1)
    
    # Combine: primarily Thompson Sampling score with a small exploration push
    # The exploration term adds a small positive score for uncertain regions
    score = ts_score + 0.05 * explore
    
    # Cost-aware: divide by evaluation cost (higher cost = lower priority)
    cost_safe = np.maximum(cost, 1e-9)
    score = score / cost_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
