"""
Optimized solution for the third autocorrelation inequality task.

Strategy: Use L-BFGS-B with a soft-max surrogate to minimize
    V = |2*N*max(convolve(h,h)) / (sum(h))**2|

Key insight: The max over the autoconvolution can be driven below the
Cauchy-Schwarz lower bound at lag 0 by using signed vectors where
positive and negative lobes cause destructive interference at the peak.
The soft-max makes the non-smooth max differentiable for gradient-based
optimization.

Sparse-spike initialization reduces effective dimensionality and helps
L-BFGS-B navigate the non-convex landscape more efficiently.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 150.0
SAFETY_MARGIN = 10.0


def _compute_value(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    conv = np.convolve(h, h)
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
    conv = np.convolve(h, h)
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
    conv = np.convolve(h, h)
    m = np.max(conv)
    if m > 1e15:
        return np.ones_like(h)
    # weights for soft max
    exp_conv = np.exp(beta * (conv - m))
    sum_exp = np.sum(exp_conv)
    if sum_exp < 1e-300:
        return np.ones_like(h)
    weights = exp_conv / sum_exp  # softmax weights over conv entries

    # d(soft_max)/d(conv_k) = weights[k]
    # d(conv_k)/d(h_i) = h_{k-i} for valid indices (full convolve)
    # conv = np.convolve(h, h), length 2N-1
    # conv[k] = sum_i h[i]*h[k-i] where 0<=i<N, 0<=k-i<N
    # d(conv[k])/d(h[j]) = sum_i [i==j]*h[k-i] + h[i]*[k-i==j]
    #                     = h[k-j] (if 0<=k-j<N) + h[k-j] (if 0<=k-j<N, same term)
    # Actually for full convolve: conv[k] = sum_{i} h[i]*h[k-i]
    # d(conv[k])/d(h[j]) = h[k-j] (from first h) + h[k-j] (from second h) if both in range
    # More carefully: d/d(h[j]) sum_i h[i]*h[k-i] = sum_i [delta_{i,j}*h[k-i] + h[i]*delta_{k-i,j}]
    # = h[k-j] + h[k-j] = 2*h[k-j] if 0<=k-j<N, else h[k-j] or h[j-k]...
    # Let's just use the fact that conv is quadratic in h.
    # Actually, let's compute gradient numerically-free via the structure:
    # conv[k] = sum_i h[i] * h[k-i]  (full convolution, k from 0 to 2N-2)
    # d(conv[k])/d(h[j]) = h[k-j] (if 0<=k-j<N) + h[k-j] (if 0<=k-j<N)
    # Wait, that's not right. Let me be careful:
    # conv[k] = sum_{i=max(0,k-N+1)}^{min(k,N-1)} h[i] * h[k-i]
    # d(conv[k])/d(h[j]) = h[k-j] (if 0<=k-j<N and j in range for first term)
    #                    + h[k-j] (if 0<=k-j<N and k-j in range for second term)
    # Since k-j is the same in both, if 0<=k-j<N then both terms contribute h[k-j].
    # But wait, the first term: d/d(h[j]) of h[i]*h[k-i] = delta_{i,j}*h[k-i] = h[k-j] if i=j
    # The second term: d/d(h[j]) of h[i]*h[k-i] = h[i]*delta_{k-i,j} = h[i] if k-i=j, i.e., i=k-j
    # So d(conv[k])/d(h[j]) = h[k-j] (if 0<=k-j<N) + h[k-j] (if 0<=k-j<N)
    # = 2*h[k-j] if 0<=k-j<N, else 0.
    # Hmm, but that's only when j is in [0,N-1] and k-j is in [0,N-1].
    # Actually it's: contribution from first factor is h[k-j] if 0<=k-j<N
    # contribution from second factor is h[k-j] if 0<=k-j<N
    # So total is 2*h[k-j] if 0<=k-j<N, else 0.

    # Build gradient of soft_max w.r.t. h
    # d(soft_max)/d(h[j]) = (1/beta) * sum_k weights[k] * d(conv[k])/d(h[j])
    # = (1/beta) * sum_k weights[k] * 2*h[k-j] (for valid k-j)

    # For efficiency, note that d(conv[k])/d(h[j]) = 2*h[k-j] for 0<=k-j<N
    # So d(soft_max)/d(h[j]) = (2/beta) * sum_{k: 0<=k-j<N} weights[k] * h[k-j]
    # Let l = k-j, so k = j+l, l in [0, N-1]
    # = (2/beta) * sum_{l=0}^{N-1} weights[j+l] * h[l]  (where j+l < 2N-1)

    # This is a convolution-like operation
    # grad_softmax[j] = (2/beta) * sum_l weights[j+l] * h[l]
    # which is like a cross-correlation

    # For the full gradient of val^2 where val = 2*N*soft_max/S^2:
    # d(val^2)/d(h[j]) = 2*val * d(val)/d(h[j])
    # d(val)/d(h[j]) = 2*N * d(soft_max)/d(h[j])/S^2 - 2*val * S / S^2 * 1
    #                = 2*N * d(soft_max)/d(h[j])/S^2 - 2*val/S

    val = 2.0 * N * (m + np.log(sum_exp)) / beta / (s ** 2)

    # Compute d(soft_max)/d(h) efficiently
    # grad_sm[j] = (2/beta) * sum_{l=0}^{min(N-1, 2N-2-j)} weights[j+l] * h[l]
    # This is a correlation: for each j, dot product of h[0:min(N,2N-1-j)] with weights[j:j+min(N,2N-1-j)]

    # Use a simple approach: for N=400, 2N-1=799, this is manageable
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


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE - SAFETY_MARGIN

    best_h = np.ones(N)
    best_val = _compute_value(best_h)  # 2.0

    # Try multiple initializations with increasing soft-max sharpness
    inits = [
        _sparse_spike_init(42),
        _sparse_spike_init(123),
        _sparse_spike_init(777),
        np.ones(N),  # seed
    ]

    # Also try a structured initialization: alternating positive/negative blocks
    h_struct = np.zeros(N)
    block_size = 20
    for i in range(N // block_size):
        sign = 1.0 if i % 2 == 0 else -1.0
        h_struct[i * block_size:(i + 1) * block_size] = sign
    inits.append(h_struct)

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
                        'maxiter': 200,
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
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
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
    if final_val <= 2.0:
        return best_h
    return best_h
