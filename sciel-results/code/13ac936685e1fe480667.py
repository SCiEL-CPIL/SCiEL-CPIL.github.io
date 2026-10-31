"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute the expected dead space metric S(r) for Weibull items
# Weibull parameters: shape k=1.5, scale lambda=10 (typical for capacity 100)
# S(r) = E[max(0, r - X)] where X ~ Weibull(k, lambda)
# This is the expected waste if we leave remainder r and the next item is X
# S(r) = integral from 0 to r of (r - x) * f(x) dx + integral from r to inf of 0 * f(x) dx
#      = r * (1 - F(r)) - integral from 0 to r x * f(x) dx
#      = r * (1 - F(r)) - (lambda^k / Gamma(k+1)) * Gamma(k+1) * (1 - (r/lambda)^k * E[X|X<r]... )
# Actually, let's use the survival function and mean residual life concept.
# E[max(0, r-X)] = ∫_0^r P(X < t) dt ... no, that's E[min(r,X)].
# E[max(0, r-X)] = ∫_0^r P(X < x) dx? No.
# Let Y = max(0, r-X). E[Y] = ∫_0^r P(Y > y) dy = ∫_0^r P(r-X > y) dy = ∫_0^r P(X < r-y) dy
# Let u = r-y, dy = -du. Limits: y=0->u=r, y=r->u=0.
# E[Y] = ∫_0^r P(X < u) du = ∫_0^r F(u) du.
# For Weibull(k, lambda), F(u) = 1 - exp(-(u/lambda)^k).
# So S(r) = ∫_0^r (1 - exp(-(u/lambda)^k)) du = r - ∫_0^r exp(-(u/lambda)^k) du.
# Let v = (u/lambda)^k => u = lambda * v^(1/k), du = (lambda/k) * v^(1/k - 1) dv.
# ∫_0^r exp(-(u/lambda)^k) du = (lambda/k) ∫_0^{(r/lambda)^k} exp(-v) v^(1/k - 1) dv = (lambda/k) * gamma(1+1/k, (r/lambda)^k)
# where gamma is the lower incomplete gamma function.

from scipy.special import gammainc, gamma

# Weibull parameters
K_WEIBULL = 1.5
LAMBDA_WEIBULL = 10.0
CAPACITY = 100.0

# Pre-compute S(r) for r in [0, 100]
# We need S(r) = r - (lambda/k) * gamma(1+1/k) * gammainc(1+1/k, (r/lambda)^k)
# Note: scipy.special.gammainc(a, x) is the regularized lower incomplete gamma P(a, x) = gamma(a, x) / Gamma(a)
# So gamma(a, x) = Gamma(a) * gammainc(a, x)
# Thus, ∫_0^r exp(...) du = (lambda/k) * Gamma(1+1/k) * gammainc(1+1/k, (r/lambda)^k)

# Let's precompute a lookup table for S(r) for r from 0 to 100 with step 0.1
_r_vals = np.arange(0, CAPACITY + 0.1, 0.1)
_a = 1.0 + 1.0 / K_WEIBULL
_b = (LAMBDA_WEIBULL / K_WEIBULL) * gamma(_a)
_S_table = _r_vals - _b * gammainc(_a, (_r_vals / LAMBDA_WEIBULL) ** K_WEIBULL)

# Function to get S(r) via interpolation
def _expected_dead_space(r):
    # Clip r to [0, 100]
    r_clipped = np.clip(r, 0, CAPACITY)
    # Linear interpolation
    idx = np.searchsorted(_r_vals, r_clipped)
    if idx == 0:
        return _S_table[0]
    if idx >= len(_r_vals):
        return _S_table[-1]
    x0, x1 = _r_vals[idx-1], _r_vals[idx]
    y0, y1 = _S_table[idx-1], _S_table[idx]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (r_clipped - x0) / (x1 - x0)

# Vectorized version for efficiency
def _expected_dead_space_vec(r_arr):
    r_clipped = np.clip(r_arr, 0, CAPACITY)
    idx = np.searchsorted(_r_vals, r_clipped)
    idx = np.clip(idx, 1, len(_r_vals) - 1)
    x0, x1 = _r_vals[idx-1], _r_vals[idx]
    y0, y1 = _S_table[idx-1], _S_table[idx]
    denom = x1 - x0
    denom[denom == 0] = 1.0
    return y0 + (y1 - y0) * (r_clipped - x0) / denom

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 150.0
_TIME_MARGIN = 5.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _START_TIME
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _TIME_MARGIN:
        # Fallback to simple best-fit if we're running out of time
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full_like(remaining, -np.inf)
        scores[valid] = -remaining[valid]
        return scores

    remaining = bins - item
    valid = remaining >= 0
    scores = np.full_like(remaining, -np.inf)
    
    if not np.any(valid):
        return scores

    valid_remaining = remaining[valid]
    
    # Compute expected dead space for each valid bin
    # We want to minimize S(remaining), so score = -S(remaining)
    s_vals = _expected_dead_space_vec(valid_remaining)
    
    # Also consider a small penalty for very large remainders to avoid "wasting" big bins
    # The S(r) metric already does this, but let's ensure it's monotonic or behaves well.
    # S(r) is increasing in r, so -S(r) is decreasing.
    # This means we prefer bins with smaller remainders, which is good.
    # But we also want to avoid leaving a remainder that is "unusable".
    # S(r) captures this.
    
    scores[valid] = -s_vals
    
    return scores
# EVOLVE-BLOCK-END
