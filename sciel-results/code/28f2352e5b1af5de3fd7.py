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
    Cost-aware batch-aware Thompson Sampling acquisition function for Bayesian optimization
    on Shekel 4D. Samples from the GP posterior and selects points that maximize
    expected improvement per unit cost relative to the current best.
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

    # For minimization, we want low values. 
    # Batch-aware Thompson Sampling heuristic:
    # Select candidate that maximizes expected improvement per unit cost.
    # EI(x) = (f_best - sampled_f) * Phi((f_best - sampled_f)/sigma) + sigma * phi((f_best - sampled_f)/sigma)
    # But for batch-aware, we use the sampled function value directly.
    # The improvement is max(0, f_best - sampled_f).
    # We want to maximize this improvement divided by cost.
    
    # Compute improvement over current best
    improvement = np.maximum(0, f_best - sampled_f)
    
    # Expected improvement per unit cost
    ei_per_cost = improvement / cost_safe
    
    # Also add a small exploration bonus for high uncertainty
    # This helps escape local optima in the sampled landscape
    exploration_bonus = 0.05 * sigma / cost_safe

    scores = ei_per_cost + exploration_bonus

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
