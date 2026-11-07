"""Rewritten acquisition for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 315.0  # safety margin vs 330s kill


def acquisition(mu, sigma, f_best):
    # Deadline guard: if we're getting close to the limit, return a safe fallback
    elapsed = time.time() - _T0
    if elapsed > _DEADLINE:
        # Return scores that favor low mu (exploitation) to be safe
        return -mu.copy()

    # Ensure sigma is positive
    s = np.maximum(sigma, 1e-8)

    # Minimum-variance acquisition: score candidates by posterior standard deviation.
    # This forces the GP to resolve uncertainty in unexplored regions,
    # preventing premature convergence to local noise minima.
    #
    # For the sphere function on [-5, 5], the global minimum is at x=0.
    # The GP with Matern-5/2 and length scale 0.2 (on [0,1] scaled coords)
    # will have high sigma in unexplored regions. By prioritizing high sigma,
    # we ensure the GP builds a good model of the entire landscape before
    # exploiting.
    #
    # However, pure minimum-variance can be too aggressive in exploration.
    # We combine it with a mild exploitation signal to avoid completely
    # ignoring regions where we already have good estimates.
    #
    # Strategy:
    # 1. Primary: score by sigma (minimum-variance / maximum-uncertainty)
    # 2. Secondary: mild penalty for very high mu (points far from f_best)
    #    to avoid exploring regions that are clearly bad.
    #
    # The combination: score = sigma * exp(-lambda * max(0, mu - f_best) / sigma)
    # This is related to the "probability of improvement" but weighted by sigma.
    #
    # Actually, let's use a cleaner approach:
    # score = sigma * (1 - norm.cdf((mu - f_best) / sigma))
    # This is sigma * P(X > f_best) where X ~ N(mu, sigma^2)
    # For minimization, we want P(X < f_best), so:
    # score = sigma * norm.cdf((f_best - mu) / sigma)
    #
    # This is the "expected maximum improvement" style, but simpler.
    # It balances exploration (high sigma) with exploitation (low mu relative to f_best).

    z = (f_best - mu) / s
    cdf_z = norm.cdf(z)
    
    # Minimum-variance with exploitation bias
    score = s * cdf_z
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
