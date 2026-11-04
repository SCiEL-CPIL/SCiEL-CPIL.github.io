"""Signed autocorrelation maximum absolute-value variant.

Parametric construction: represent the sequence as a sum of a small number of
shifted, signed block pulses (Dirichlet-like rectangular kernels) with
continuous positions and widths. Optimize these geometric parameters via
differential evolution to minimize the maximum absolute autoconvolution
divided by the squared integral.
"""

import time
import numpy as np
from scipy.optimize import differential_evolution

N = 400
TIME_LIMIT = 120  # seconds; leave margin for overhead
DEADLINE = time.time() + TIME_LIMIT - 10  # 10 s safety margin


def _autoconv_metric(x):
    """Compute max(|autoconv(x)|) / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    # Full linear autoconvolution via FFT
    n = len(x)
    size = 2 * n
    X = np.fft.rfft(x, n=size)
    ac = np.fft.irfft(X * X, n=size)[:2 * n - 1]
    return np.max(np.abs(ac)) / (s * s)


def _build_from_params(params, n_blocks=8):
    """Build a signed step sequence of length N from block-pulse parameters.
    
    params: array of shape (n_blocks * 3,) where each triple is
            (start_frac, width_frac, amplitude).
            start_frac in [0, 1), width_frac in [0.05, 0.5], amplitude in [-1, 1].
    """
    x = np.zeros(N)
    for i in range(n_blocks):
        start_frac = params[3 * i]
        width_frac = params[3 * i + 1]
        amp = params[3 * i + 2]
        start = int(start_frac * N)
        width = max(1, int(width_frac * N))
        end = min(N, start + width)
        x[start:end] += amp
    return x


def _objective(params):
    """Objective for differential evolution: the autoconvolution metric."""
    x = _build_from_params(params)
    return _autoconv_metric(x)


def construct():
    """Return a sequence of signed step heights (length N) minimizing the
    maximum absolute autoconvolution divided by squared integral."""
    n_blocks = 8
    n_params = n_blocks * 3

    # Bounds for each parameter
    bounds = []
    for _ in range(n_blocks):
        bounds.append((0.0, 1.0))      # start_frac
        bounds.append((0.05, 0.5))     # width_frac
        bounds.append((-1.0, 1.0))     # amplitude

    # Initial best: the seed (all ones)
    x_seed = np.ones(N)
    best_x = x_seed.copy()
    best_val = _autoconv_metric(x_seed)

    # Differential evolution with time budget
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=True,
            workers=-1,
            callback=lambda p, con: None,
        )
        if result.success or result.fun < best_val:
            x_candidate = _build_from_params(result.x)
            val = _autoconv_metric(x_candidate)
            if val < best_val:
                best_x = x_candidate
                best_val = val
    except Exception:
        pass

    # If time remains, try a second round with different block count
    if time.time() < DEADLINE:
        n_blocks2 = 6
        n_params2 = n_blocks2 * 3
        bounds2 = []
        for _ in range(n_blocks2):
            bounds2.append((0.0, 1.0))
            bounds2.append((0.05, 0.5))
            bounds2.append((-1.0, 1.0))
        try:
            result2 = differential_evolution(
                _objective,
                bounds=bounds2,
                maxiter=30,
                popsize=12,
                tol=1e-6,
                seed=123,
                polish=True,
            )
            if result2.fun < best_val:
                x_candidate = _build_from_params(result2.x, n_blocks=n_blocks2)
                val = _autoconv_metric(x_candidate)
                if val < best_val:
                    best_x = x_candidate
                    best_val = val
        except Exception:
            pass

    # Final local refinement via coordinate descent on the best discrete sequence
    if time.time() < DEADLINE:
        x = best_x.copy()
        # Try flipping individual elements to see if it helps
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for i in range(0, N, 4):  # step by 4 for speed
                old_val = _autoconv_metric(x)
                x[i] = -x[i]
                new_val = _autoconv_metric(x)
                if new_val < old_val - 1e-10:
                    improved = True
                else:
                    x[i] = -x[i]  # revert
            # Also try small shifts of blocks
            if improved and time.time() < DEADLINE:
                for i in range(0, N, 8):
                    old_val = _autoconv_metric(x)
                    # Shift element i to i+1
                    if i + 1 < N:
                        x[i + 1] += x[i]
                        x[i] = 0
                        new_val = _autoconv_metric(x)
                        if new_val < old_val - 1e-10:
                            improved = True
                        else:
                            x[i] = x[i + 1]
                            x[i + 1] = 0
    # Ensure non-trivial (not all zeros)
    if np.allclose(x, 0):
        x = best_x if np.allclose(best_x, 0) else best_x
    if np.allclose(x, 0):
        x = np.ones(N)

    return x
