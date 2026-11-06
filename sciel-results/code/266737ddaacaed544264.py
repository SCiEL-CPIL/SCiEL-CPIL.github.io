"""Flattest plus-minus-one 24-coefficient polynomial search.

Strategy:
1. Start from Rudin-Shapiro seed.
2. Use simulated annealing with fast FFT-based peak evaluation.
3. Polish the best candidate with high-resolution local search.
"""
import math
import time
import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 120.0


def _time_left():
    return _LIMIT - (time.time() - _T0)


def _rs_poly():
    """Rudin-Shapiro coefficients truncated to _N."""
    p, q = [1], [1]
    while len(p) < _N:
        p, q = p + q, p + [-c for c in q]
    return p[:_N]


def _peak_fft(coeffs, M=2048):
    """Compute peak |P| on unit circle using FFT-based dense sampling."""
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    z = np.exp(1j * theta)
    c = np.array(coeffs, dtype=complex)
    powers = np.arange(_N)
    # Z[k,j] = z_k^j
    Z = np.power(z[:, None], powers[None, :])
    vals = Z @ c
    mag2 = np.abs(vals) ** 2
    best_idx = np.argmax(mag2)
    return math.sqrt(mag2[best_idx]), theta[best_idx]


def _peak_refine(coeffs, phase, M, iters=15):
    """Golden-section refinement of local peak around phase."""
    lo = phase - math.pi / M
    hi = phase + math.pi / M
    c_arr = np.array(coeffs, dtype=complex)
    powers = np.arange(_N)

    def f(a):
        z = np.exp(1j * a)
        zk = np.power(z, powers)
        s = np.dot(c_arr, zk)
        return s.real * s.real + s.imag * s.imag

    # Ternary search for maximum
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = f(m1)
        f2 = f(m2)
        if f1 > f2:
            lo = m1
        else:
            hi = m2
    mid = (lo + hi) / 2.0
    return math.sqrt(f(mid))


def _score_final(coeffs, M=4096):
    """Final high-precision score."""
    peak, phase = _peak_fft(coeffs, M)
    # Refine top few peaks
    nb = max(1, M // 1024)
    for d in (0, nb, -nb, 2*nb, -2*nb):
        ph = phase + 2.0 * math.pi * d / M
        p = _peak_refine(coeffs, ph, M, iters=20)
        if p > peak:
            peak = p
    return peak / _SQRTN


def _simulate_annealing(seed, deadline):
    """Simulated annealing to find low-peak coefficient sequence."""
    cur = list(seed)
    cur_peak, cur_phase = _peak_fft(cur, M=1024)
    best = list(cur)
    best_peak = cur_peak

    T0 = 1.0
    T = T0
    cooling = 0.999
    rng = np.random.default_rng(12345)

    while _time_left() > 20.0:
        # Flip a random coefficient
        j = int(rng.integers(0, _N))
        cur[j] = -cur[j]
        new_peak, new_phase = _peak_fft(cur, M=1024)
        delta = new_peak - cur_peak

        if delta < 0 or rng.random() < math.exp(-delta / (T * 0.1)):
            cur_peak = new_peak
            cur_phase = new_phase
            if cur_peak < best_peak - 1e-12:
                best = list(cur)
                best_peak = cur_peak
        else:
            cur[j] = -cur[j]  # revert

        T *= cooling
        if T < 0.01:
            T = 0.01

    return best


def _local_search_polish(seed, deadline):
    """High-resolution local search polish."""
    cur = list(seed)
    cur_peak, cur_phase = _peak_fft(cur, M=2048)
    best = list(cur)
    best_peak = cur_peak

    # Refine current peak
    cur_peak = _peak_refine(cur, cur_phase, 2048, iters=25)
    best_peak = cur_peak

    # Try all single flips with high-resolution measurement
    for j in range(_N):
        if _time_left() < 5.0:
            break
        cur[j] = -cur[j]
        p, ph = _peak_fft(cur, M=2048)
        p = _peak_refine(cur, ph, 2048, iters=25)
        if p < best_peak - 1e-12:
            best_peak = p
            best = list(cur)
        else:
            cur[j] = -cur[j]

    # Second pass
    if _time_left() > 5.0:
        cur = list(best)
        for j in range(_N):
            if _time_left() < 3.0:
                break
            cur[j] = -cur[j]
            p, ph = _peak_fft(cur, M=2048)
            p = _peak_refine(cur, ph, 2048, iters=25)
            if p < best_peak - 1e-12:
                best_peak = p
                best = list(cur)
            else:
                cur[j] = -cur[j]

    return best


def construct():
    global _T0
    _T0 = time.time()

    # Seed: Rudin-Shapiro
    seed = _rs_poly()

    # Phase 1: Simulated annealing with moderate resolution
    best = _simulate_annealing(seed, _LIMIT)

    # Phase 2: High-resolution local search polish
    if _time_left() > 15.0:
        best = _local_search_polish(best, _LIMIT)

    # Ensure valid output
    best = [1 if v >= 0 else -1 for v in best]
    return best[:_N]
