"""Optimize signed step heights (400 intervals) to minimize a C3 autoconvolution upper bound."""
import numpy as np
import time

N = 400
_DEADLINE = time.time() + 110.0  # safety margin under 120s / 150s kill


def _autocorr_c3_bound(h):
    """
    Return a proxy for the upper bound on C3 (third autocorrelation moment).
    h: array of N signed step heights.
    Uses the normalized third moment of the (circular) autocorrelation.
    Lower is better.
    """
    n = len(h)
    # linear autocorrelation via FFT
    size = 1
    while size < 2 * n:
        size <<= 1
    F = np.fft.rfft(h, size)
    ac = np.fft.irfft(F * np.conj(F), size)[:n]
    # ac[k] = sum_i h[i]*h[i+k], k>=0
    # full (symmetric) autocorrelation sum of cubes:
    # R[0] = ac[0]; for k>0, R[k]=R[-k]=ac[k]
    val = ac[0] ** 3
    for k in range(1, n):
        val += 2.0 * ac[k] ** 3
    # normalize so that a "perfect" sequence (only lag 0) gives ~1.0
    return val / (n ** 3)


def _seed_value():
    h = np.ones(N)
    return _autocorr_c3_bound(h)


def construct():
    rng = np.random.default_rng(12345)
    # start from a random +/-1 sequence
    h = np.where(rng.random(N) < 0.5, 1.0, -1.0)
    best_h = h.copy()
    best_val = _autocorr_c3_bound(best_h)

    # also consider all-ones as a candidate (the seed)
    ones_val = _autocorr_c3_bound(np.ones(N))
    if ones_val < best_val:
        best_h = np.ones(N)
        best_val = ones_val

    # Simulated annealing / coordinate descent over single-bit flips
    # Objective is cubic in autocorrelation; use FFT recompute each flip.
    n = N
    T = 5.0
    cooling = 0.9995
    step = 0
    while time.time() < _DEADLINE:
        i = rng.integers(0, n)
        h[i] = -h[i]
        new_val = _autocorr_c3_bound(h)
        delta = new_val - best_val
        if new_val < best_val or rng.random() < np.exp(-max(0.0, delta) / max(T, 1e-9)):
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            h[i] = -h[i]  # revert
        step += 1
        if step % 200 == 0:
            T *= cooling
            if T < 1e-3:
                T = 1e-3

    return best_h
