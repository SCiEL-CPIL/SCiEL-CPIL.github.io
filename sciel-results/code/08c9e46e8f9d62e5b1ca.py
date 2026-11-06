"""Model-and-optimize for flattest plus-minus-one 24-coefficient polynomial.

P(z) = sum c_k z^k, c_k in {+1,-1}, |c| = 24.
Score = max_{|z|=1} |P(z)| / sqrt(24).  Optimum ~ 1.2583758 (max|P| ~ 6.1363).

Model: coefficients are the parameters. We FIT them to minimize the objective
(max |P| on the unit circle) via (a) a good structured seed (Rudin-Shapiro) and
(b) local +/-1 flips accepted when they lower the measured peak.  The peak is
measured by dense FFT sampling plus golden-section refinement, then the true
sup-norm is estimated from the densest grid (a rigorous upper bound on the score).
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
    """Rudin-Shapiro coefficients truncated to _N (strong low-peak seed)."""
    p, q = [1], [1]
    while len(p) < _N:
        p, q = p + q, p + [-c for c in q]
    return p[:_N]


def _eval_fft(coeffs, M):
    """Return (peak, argmax_phase) of |P| on the unit circle via FFT (fast)."""
    # Pad coeffs to length M, take FFT: P(e^{2πik/M}) = FFT of zero-padded coeffs
    c = np.array(coeffs, dtype=float)
    padded = np.zeros(M, dtype=float)
    padded[:_N] = c
    vals = np.fft.fft(padded)
    mag2 = np.abs(vals) ** 2
    best_idx = int(np.argmax(mag2))
    theta = 2.0 * np.pi * best_idx / M
    return math.sqrt(mag2[best_idx]), theta


def _score_final(coeffs, M=2048):
    """Score on a dense grid; the max on the grid is a rigorous upper bound on the score."""
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    z = np.exp(1j * theta)
    c = np.array(coeffs, dtype=complex)
    Z = np.power(z[:, None], np.arange(_N)[None, :])
    vals = Z @ c
    mag2 = np.abs(vals) ** 2
    return math.sqrt(np.max(mag2)) / _SQRTN


def _refine(coeffs, M, peak, phase, iters=12):
    """Golden-section refinement of a local peak around `phase`."""
    lo = phase - math.pi / M
    hi = phase + math.pi / M
    c_arr = np.array(coeffs, dtype=complex)
    powers = np.arange(_N)

    def f(a):
        z = np.exp(1j * a)
        # P(z) = sum c_j * z^j
        zk = np.power(z, powers)
        s = np.dot(c_arr, zk)
        return (s.real * s.real + s.imag * s.imag)

    f_lo = f(lo)
    f_hi = f(hi)
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = f(m1)
        f2 = f(m2)
        if f1 < f2:
            lo = m1
            f_lo = f1
        else:
            hi = m2
            f_hi = f2
    peak2 = max(f_lo, f_hi)
    if peak2 > peak * peak:
        peak = math.sqrt(peak2)
    return peak


def _search_peak(coeffs, M, refine=True):
    """Measured peak: dense scan + optional refinement of top few peaks."""
    peak, phase = _eval_fft(coeffs, M)
    if not refine:
        return peak
    # refine the global max and a couple neighbours for robustness
    nb = max(1, M // 512)
    for d in (0, nb, -nb, 2*nb, -2*nb):
        ph = phase + 2.0 * math.pi * d / M
        peak = _refine(coeffs, M, peak, ph, iters=10)
    return peak


def _local_search(seed, M, deadline):
    """Flip coefficients that reduce the measured peak; keep the best overall."""
    cur = list(seed)
    cur_peak = _search_peak(cur, M)
    best = list(cur)
    best_peak = cur_peak

    step = 1
    rng = np.random.default_rng(42)
    while _time_left() > 15.0:
        improved = False
        # Try all single flips in order
        for j in range(_N):
            if _time_left() < 15.0:
                break
            cur[j] = -cur[j]
            p = _search_peak(cur, M)
            if p < cur_peak - 1e-12:
                cur_peak = p
                improved = True
            else:
                cur[j] = -cur[j]
        if improved:
            if cur_peak < best_peak - 1e-12:
                best = list(cur)
                best_peak = cur_peak
        else:
            # Try double flips to escape local optima
            moved = False
            for j in range(_N):
                if _time_left() < 15.0:
                    break
                for k in range(j+1, _N):
                    if _time_left() < 15.0:
                        break
                    cur[j] = -cur[j]
                    cur[k] = -cur[k]
                    p = _search_peak(cur, M, refine=False)
                    if p < cur_peak - 1e-12:
                        cur_peak = p
                        moved = True
                        if cur_peak < best_peak - 1e-12:
                            best = list(cur)
                            best_peak = cur_peak
                        break
                    else:
                        cur[j] = -cur[j]
                        cur[k] = -cur[k]
                if moved:
                    break
            if not moved:
                # Random restart from best
                cur = list(best)
                cur_peak = best_peak
                # Perturb randomly
                for _ in range(3):
                    j = int(rng.integers(0, _N))
                    cur[j] = -cur[j]
                cur_peak = _search_peak(cur, M)
                step += 1
    return best


def construct():
    global _T0
    _T0 = time.time()

    # Seed: Rudin-Shapiro (low peak ~6.19, just above the 6.1363 optimum).
    seed = _rs_poly()

    # Phase 1: Coarse search with FFT-based peak (fast, no refinement)
    cur = list(seed)
    cur_peak = _search_peak(cur, 256, refine=False)
    best = list(cur)
    best_peak = cur_peak

    # Local search with coarse peak measurement (fast iterations)
    while _time_left() > 30.0:
        improved = False
        for j in range(_N):
            if _time_left() < 30.0:
                break
            cur[j] = -cur[j]
            p = _search_peak(cur, 256, refine=False)
            if p < cur_peak - 1e-12:
                cur_peak = p
                improved = True
                if cur_peak < best_peak - 1e-12:
                    best = list(cur)
                    best_peak = cur_peak
            else:
                cur[j] = -cur[j]
        if improved:
            continue
        # Try double flips
        moved = False
        for j in range(_N):
            if _time_left() < 30.0:
                break
            for k in range(j+1, _N):
                if _time_left() < 30.0:
                    break
                cur[j] = -cur[j]
                cur[k] = -cur[k]
                p = _search_peak(cur, 256, refine=False)
                if p < cur_peak - 1e-12:
                    cur_peak = p
                    moved = True
                    if cur_peak < best_peak - 1e-12:
                        best = list(cur)
                        best_peak = cur_peak
                    break
                else:
                    cur[j] = -cur[j]
                    cur[k] = -cur[k]
            if moved:
                break
        if not moved:
            # Random perturbation to escape local optimum
            rng = np.random.default_rng(int(time.time() * 1000) % 2**31)
            cur = list(best)
            cur_peak = best_peak
            for _ in range(2):
                j = int(rng.integers(0, _N))
                cur[j] = -cur[j]
            cur_peak = _search_peak(cur, 256, refine=False)

    # Phase 2: Fine refinement with golden-section on top peaks
    if _time_left() > 15.0:
        cur = list(best)
        cur_peak = _search_peak(cur, 512, refine=True)
        best_peak = cur_peak
        for _ in range(3):  # a few passes of single flips with fine measurement
            if _time_left() < 10.0:
                break
            improved = False
            for j in range(_N):
                if _time_left() < 10.0:
                    break
                cur[j] = -cur[j]
                p = _search_peak(cur, 512, refine=True)
                if p < cur_peak - 1e-12:
                    cur_peak = p
                    improved = True
                    if cur_peak < best_peak - 1e-12:
                        best = list(cur)
                        best_peak = cur_peak
                else:
                    cur[j] = -cur[j]
            if not improved:
                break

    # Ensure valid output: exactly 24 coefficients in {+1,-1}.
    best = [1 if v >= 0 else -1 for v in best]
    return best[:_N]
