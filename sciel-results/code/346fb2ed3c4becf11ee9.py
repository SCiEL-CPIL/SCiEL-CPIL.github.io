"""
Optimized solution for the third autocorrelation inequality task.

Strategy: Use L-BFGS-B with a soft-max surrogate to minimize
    V = |2*N*max(convolve(h,h)) / (sum(h))**2|

Key insight: The max over the autoconvolution can be driven below the
Cauchy-Schwarz lower bound at lag 0 by using signed vectors where
positive and negative lobes cause destructive interference at the peak.
The soft-max makes the non-smooth max differentiable for gradient-based
optimization.

FFT-based autoconvolution speeds up evaluation significantly.
Multiple initialization strategies and adaptive beta values are used.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import fft, ifft

N = 400
DEADLINE = 150.0
SAFETY_MARGIN = 15.0


def _autoconv_fft(h):
    """Compute full autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    # Use next power of 2 for FFT efficiency
    fft_size = 1 << (size - 1).bit_length()
    H = fft(h, n=fft_size)
    conv = np.real(ifft(H * H, n=fft_size))[:size]
    return conv


def _compute_value(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    conv = _autoconv_fft(h)
    return abs(2.0 * N * np.max(conv) / (s ** 2))


def _soft_max_obj(x, beta):
    """
    Soft-max surrogate objective.
    x is the flat vector of heights.
    We minimize a smoothed version of |2*N*max(conv)/S^2|
    using log-sum-exp for the max.
    """
    h = x
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    conv = _autoconv_fft(h)
    # Soft max: (1/beta)*log(sum(exp(beta*c))) approximates max(c)
    # Use stable log-sum-exp
    m = np.max(conv)
    if m > 1e15:
        return 1e10
    logsum = m + np.log(np.sum(np.exp(beta * (conv - m))))
    soft_max_val = logsum / beta
    val = 2.0 * N * soft_max_val / (s ** 2)
    # We want to minimize |val|, but we can work with val directly
    # and let the optimizer find negative peaks
    return val ** 2  # smooth, always positive, minimizes |val|


def _soft_max_grad(x, beta):
    """Gradient of the soft-max squared objective."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.ones_like(h)
    conv = _autoconv_fft(h)
    m = np.max(conv)
    if m > 1e15:
        return np.ones_like(h)
    # weights for soft max
    exp_conv = np.exp(beta * (conv - m))
    sum_exp = np.sum(exp_conv)
    if sum_exp < 1e-300:
        return np.ones_like(h)
    weights = exp_conv / sum_exp  # softmax weights over conv entries

    # d(soft_max)/d(h[j]) = (1/beta) * sum_k weights[k] * d(conv[k])/d(h[j])
    # d(conv[k])/d(h[j]) = 2*h[k-j] if 0<=k-j<N, else 0
    # So d(soft_max)/d(h[j]) = (2/beta) * sum_{l=0}^{N-1} weights[j+l] * h[l]

    val = 2.0 * N * (m + np.log(sum_exp)) / beta / (s ** 2)

    # Compute d(soft_max)/d(h) efficiently using correlation
    grad_sm = np.zeros(N)
    for j in range(N):
        max_l = min(N, 2 * N - 1 - j)
        if max_l > 0:
            grad_sm[j] = (2.0 / beta) * np.dot(h[:max_l], weights[j:j + max_l])

    d_val_d_h = 2.0 * N * grad_sm / (s ** 2) - 2.0 * val / s
    grad = 2.0 * val * d_val_d_h
    return grad


def _sparse_spike_init(seed=42):
    """Create a sparse spike initialization with positive and negative lobes."""
    rng = np.random.RandomState(seed)
    h = np.zeros(N)
    # Place a few positive and negative spikes
    n_spikes = 8
    positions = rng.choice(N, size=n_spikes, replace=False)
    for i, pos in enumerate(positions):
        sign = 1.0 if i % 2 == 0 else -1.0
        h[pos] = sign * rng.uniform(0.5, 2.0)
    # Add small background
    h += rng.uniform(-0.1, 0.1, size=N)
    return h


def _random_walk_init(seed=42, drift=-0.01):
    """Create a random walk initialization with negative drift for zero-sum tendency."""
    rng = np.random.RandomState(seed)
    h = np.zeros(N)
    h[0] = 0
    for i in range(1, N):
        h[i] = h[i-1] + drift + rng.normal(0, 0.1)
    # Normalize to have sum close to 1
    s = np.sum(h)
    if abs(s) > 1e-10:
        h = h / s
    return h


def _structured_init(block_size=20):
    """Create structured alternating block initialization."""
    h = np.zeros(N)
    for i in range(N // block_size):
        sign = 1.0 if i % 2 == 0 else -1.0
        h[i * block_size:(i + 1) * block_size] = sign
    return h


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE - SAFETY_MARGIN

    best_h = np.ones(N)
    best_val = _compute_value(best_h)  # 2.0

    # Try multiple initializations
    inits = [
        _sparse_spike_init(42),
        _sparse_spike_init(123),
        _sparse_spike_init(777),
        _random_walk_init(42),
        _random_walk_init(99),
        _structured_init(20),
        _structured_init(10),
        np.ones(N),  # seed
    ]

    # Try a few different beta values (soft-max sharpness)
    betas = [5.0, 20.0, 50.0, 100.0]

    for beta in betas:
        if time.time() > deadline:
            break
        for init in inits:
            if time.time() > deadline:
                break
            try:
                res = minimize(
                    _soft_max_obj,
                    init.copy(),
                    jac=_soft_max_grad,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={
                        'maxiter': 300,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 50,
                    }
                )
                if res.success or res.fun < 1e10:
                    val = _compute_value(res.x)
                    if val < best_val:
                        best_val = val
                        best_h = res.x.copy()
            except Exception:
                pass

    # If we have time, do a final polish with the exact objective using a
    # local search around the best point found
    if time.time() < deadline - 5.0:
        try:
            # Use the exact value as objective with a simple local optimization
            def exact_obj(x):
                v = _compute_value(x)
                return v

            res2 = minimize(
                exact_obj,
                best_h.copy(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            val2 = _compute_value(res2.x)
            if val2 < best_val:
                best_val = val2
                best_h = res2.x.copy()
        except Exception:
            pass

    # Final validation
    final_val = _compute_value(best_h)
    if not np.isfinite(final_val) or abs(np.sum(best_h)) < 1e-15:
        return np.ones(N)

    # Return the best found
    return best_h
