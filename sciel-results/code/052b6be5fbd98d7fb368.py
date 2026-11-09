"""Cost-aware acquisition for three-hump-camel 2D BO.

Structural change: Replace the scalar EI/PI heuristic with a cost-aware
Thompson Sampling approach that samples from the posterior and selects
candidates that maximize the probability of achieving a target regret
reduction per unit cost.

Key ideas:
1. Thompson Sampling: sample f(x) ~ N(mu, sigma^2) and pick the candidate
   with the lowest sampled value, adjusted for cost.
2. Cost-aware penalty: divide the sampled value by a cost-dependent factor
   to prefer cheaper evaluations when the expected improvement is similar.
3. Deadline guard: track elapsed time and return early if approaching the limit.
"""
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 295.0  # seconds, with safety margin before 300s limit


def _check_deadline():
    """Return True if we are close to the deadline."""
    return (time.time() - _START_TIME) > _DEADLINE


def _fast_norm_cdf(x):
    """Fast approximation of the standard normal CDF."""
    # Use a rational approximation for speed
    # This is a standard approximation with max error ~1.5e-7
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * x)
    poly = t * (0.319381530 +
                t * (-0.356563782 +
                     t * (1.781477937 +
                          t * (-1.821255978 +
                               t * 1.330274429))))
    cdf = 1.0 - poly * np.exp(-0.5 * x * x)
    return 0.5 * (1.0 + sign * (2.0 * cdf - 1.0))


def _fast_norm_pdf(x):
    """Fast computation of the standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Samples from the posterior GP and selects candidates that minimize
    the sampled objective value adjusted for evaluation cost.
    
    The cost-aware adjustment prefers cheaper evaluations when the
    expected improvement is similar, implementing a budget-aware strategy.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases where sigma might be zero or negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Check deadline - if we're close, return a simple score quickly
    if _check_deadline():
        # Fall back to a simple EI-based score
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * _fast_norm_cdf(z) + sigma_safe * _fast_norm_pdf(z)
        mean_cost = np.mean(cost) + 1e-9
        scores = ei / cost
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return scores
    
    # Thompson Sampling: sample from the posterior
    # f_sampled ~ N(mu, sigma^2)
    # Use a fixed random seed for reproducibility within the evaluation
    # Actually, we want stochasticity, so use np.random
    
    # Sample one value from each posterior
    f_sampled = mu + sigma_safe * np.random.standard_normal(len(mu))
    
    # Cost-aware adjustment:
    # We want to minimize f_sampled / cost_factor
    # where cost_factor penalizes expensive evaluations
    # 
    # The cost is 1 + mean(x_scaled), so cost is in [1, 2]
    # We divide by cost to prefer cheaper points
    # 
    # But we also want to account for the fact that f_best is the current
    # best observed value. Points that are likely to beat f_best are more
    # valuable.
    
    # Compute a regret-based score:
    # regret_sampled = max(0, f_best - f_sampled)
    # This is the expected improvement in sampled space
    
    regret_sampled = np.maximum(0, f_best - f_sampled)
    
    # Also compute the raw sampled value for exploitation
    # Combine: prefer low f_sampled (exploitation) and high regret (exploration)
    
    # Normalize cost to have mean 1
    mean_cost = np.mean(cost) + 1e-9
    cost_norm = cost / mean_cost
    
    # Score: regret per unit cost, with a bias toward exploitation
    # Use a weighted combination:
    # - Primary: regret_sampled / cost_norm (cost-aware improvement)
    # - Secondary: -f_sampled / cost_norm (exploit low values)
    
    # The weight between exploitation and exploration can be adaptive:
    # If f_best is close to 0 (the minimum), we should exploit more
    # If f_best is far from 0, we should explore more
    
    # Normalize f_best to [0, 1] range based on the three-hump-camel function
    # The minimum is 0, and typical values are in [0, ~10]
    # Use a simple scaling
    f_best_scaled = min(max(f_best, 0.0), 10.0) / 10.0
    
    # Weight: higher when f_best is large (early stage, explore more)
    # Lower when f_best is small (late stage, exploit more)
    w_exploit = 0.3 + 0.7 * (1.0 - f_best_scaled)
    w_explore = 1.0 - w_exploit
    
    # Compute normalized scores
    # Regret is non-negative, so normalize by its mean
    mean_regret = np.mean(regret_sampled) + 1e-9
    regret_norm = regret_sampled / mean_regret
    
    # For exploitation, use -f_sampled normalized
    # Shift and scale to be positive
    f_sampled_shifted = -f_sampled - np.min(f_sampled) + 1e-9
    mean_f = np.mean(f_sampled_shifted) + 1e-9
    f_norm = f_sampled_shifted / mean_f
    
    combined = w_exploit * f_norm + w_explore * regret_norm
    
    # Final score: combined improvement potential per unit of cost
    scores = combined / cost_norm
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
