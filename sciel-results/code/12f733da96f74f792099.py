import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0
_total_evals = 0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function for Bayesian optimization
    on Shekel 4D. Uses expected improvement per unit cost as the scoring rule.
    """
    global _start_time, _call_count, _total_evals

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
    # Use time-based seed for better diversity
    rng = np.random.default_rng(int(time.time() * 1e6) % (2**32))
    noise = rng.standard_normal(len(mu))

    # Thompson sampling: sample from posterior f(x) ~ N(mu, sigma)
    sampled_f = mu + sigma * noise

    # Expected improvement for minimization: EI = E[max(f_best - f(x), 0)]
    # For a sampled value s, the "improvement" is max(f_best - s, 0)
    # But for cost-awareness, we want improvement per unit cost
    # 
    # A smoother alternative: use a soft-max approximation
    # soft_max(a, b) ≈ (1/t) * log(exp(t*a) + exp(t*b)) for large t
    # Here we want soft_min(f_best, sampled_f) = -soft_max(-f_best, -sampled_f)
    #
    # Let's use: score = (f_best - sampled_f) / cost, but smoothed
    # to avoid the kink at f_best = sampled_f
    
    # Temperature for soft-min (higher = smoother)
    # Scale temperature based on typical sigma
    sigma_mean = np.mean(sigma)
    t = 1.0 / max(sigma_mean * 0.1, 1e-6)
    
    # Soft improvement: smooth approximation of max(f_best - sampled_f, 0)
    # Using log-sum-exp trick:
    # max(a, 0) ≈ (1/t) * log(1 + exp(t*a))
    # where a = f_best - sampled_f
    a = f_best - sampled_f
    # Numerically stable log(1 + exp(t*a))
    ta = t * a
    # For ta > 0: log(1 + exp(ta)) ≈ ta + log(1 + exp(-ta)) ≈ ta
    # For ta <= 0: log(1 + exp(ta)) is stable directly
    soft_improvement = np.where(
        ta > 0,
        ta + np.log1p(np.exp(-ta)),
        np.log1p(np.exp(ta))
    ) / t
    
    # Cost-aware score: improvement per unit cost
    # Higher score = better (more improvement per cost)
    scores = soft_improvement / cost_safe

    # Add a small exploration bonus scaled by uncertainty
    # This encourages evaluating uncertain points, weighted by cost efficiency
    exploration_bonus = 0.02 * sigma / cost_safe

    scores = scores + exploration_bonus

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
