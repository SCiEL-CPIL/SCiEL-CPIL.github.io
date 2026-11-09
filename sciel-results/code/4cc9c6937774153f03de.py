import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0
_total_evals = 0
# Cache for Thompson sampling noise to maintain consistency within a single acquisition call
_ts_noise_cache = {}


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function for Bayesian optimization
    on Shekel 4D. Samples from the GP posterior and selects points that minimize
    the sampled function, weighted by cost efficiency.
    """
    global _start_time, _call_count, _total_evals, _ts_noise_cache

    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    _total_evals += 1

    elapsed = time.time() - _start_time
    if elapsed > 310:
        # Safety fallback: simple cost-aware exploitation
        return -mu / np.maximum(cost, 1e-9)

    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-12, 1e6)
    cost_safe = np.maximum(cost, 1e-9)

    # Generate a single Gaussian noise sample for Thompson sampling
    # Use a deterministic seed based on call count for reproducibility within the run
    rng = np.random.default_rng(hash(_call_count) % (2**32))
    noise = rng.standard_normal(len(mu))

    # Thompson sampling: sample from posterior f(x) ~ N(mu, sigma)
    # The sampled value is mu + sigma * noise
    sampled_f = mu + sigma * noise

    # For minimization, we want low values. Thompson sampling score:
    # We want to maximize the probability that this point is good AND cheap.
    # A common approach: score = -sampled_f / cost (higher is better)
    # But this can be noisy. Let's use a more robust formulation.
    
    # Alternative: Use the expected value of the minimum of (sampled_f, f_best)
    # or simply use -sampled_f normalized by cost.
    
    # To make it more stable, we can blend with the mean:
    # score = -alpha * sampled_f - (1-alpha) * mu, all divided by cost
    # In early stages, alpha closer to 1 (more exploration via sampling)
    # In later stages, alpha closer to 0 (more exploitation via mean)
    
    if _total_evals <= 15:
        alpha = 0.8
    elif _total_evals <= 30:
        alpha = 0.6
    elif _total_evals <= 50:
        alpha = 0.4
    else:
        alpha = 0.3

    # Compute Thompson sampling score
    ts_score = -(alpha * sampled_f + (1 - alpha) * mu) / cost_safe

    # Add a small exploration bonus for high uncertainty (sigma)
    # This helps escape local optima in the sampled landscape
    exploration_bonus = 0.05 * sigma / cost_safe

    scores = ts_score + exploration_bonus

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
