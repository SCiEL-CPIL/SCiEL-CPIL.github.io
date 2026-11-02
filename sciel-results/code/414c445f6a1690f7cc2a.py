"""Signed step functions minimizing max |autoconv| / (sum)^2.
Discrete local search over the sign hypercube with FFT autoconvolution.
"""
import time
import numpy as np

N = 400
_DEADLINE = 150.0
_START = time.time()


def _left():
    return _DEADLINE - 2.0 - (time.time() - _START)  # 2s safety margin


def _autoconv(s):
    n = s.shape[0]
    m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(s, m)
    ac = np.fft.irfft(f * np.conj(f), m)[:2 * n - 1]
    return ac


def _score(s):
    ac = _autoconv(s)
    num = float(np.max(np.abs(ac)))
    den = float(np.sum(s)) ** 2
    if den <= 0.0:
        return float('inf')
    return num / den


def _round_to_signed(s):
    s = np.asarray(s, dtype=np.float64)
    return np.where(s >= 0.0, 1.0, -1.0)


def _greedy(s, budget):
    """Flip single signs (or pairs) that reduce the objective."""
    s = s.copy()
    cur = _score(s)
    n = s.shape[0]
    while _left() > budget:
        ac = _autoconv(s)
        ab = np.abs(ac)
        num = float(np.max(ab))
        den = float(np.sum(s)) ** 2
        if den <= 0.0:
            break
        # candidate: flip each i -> new num = max(|ac + 2 s_i * (s * s)|)
        # (s * s) is the autoconvolution of the sign pattern (full, length 2n-1)
        ss = ac  # autoconv of s (same as s*s autocorr)
        # effect of flipping i: ac' = ac + 2 s_i * (s convolved with delta_i)
        # delta_i convolved with s = s shifted by i (both directions)
        # We approximate by sampling a few candidate indices (the peaks).
        # Efficient: for a given i, contribution vector c_i has nonzero only near
        # positions related to i. We just evaluate top-k candidate flips.
        # Pick candidate indices: where |s| is ... all 1. Use the positions that
        # contribute most to the current max.
        # Simple: try flipping indices at the current argmax region of |ac|.
        peaks = np.argsort(ab)[::-1]
        tried = set()
        best_gain = 0.0
        best_i = None
        for i in peaks[:64]:
            if i in tried:
                continue
            tried.add(i)
            # c_i: autoconv change from flipping s_i
            # ac' = ac + 2*s_i*( s * delta_i full )
            # (s * delta_i) autoconv-like: vector v with v[j] = s_{j - i} when in range
            # We compute via: shift s by i and add reversed shift
            v = np.zeros(2 * n - 1)
            a = np.arange(0, n)
            # s_k at position (k - i) in first half ... build via index
            # contribution = 2 * s_i * ( s * delta_i ) where (s*delta_i)[t] = s[t-i]
            # autoconv of s*delta_i at lag L = sum_t s[t-i] s[t-i + L]
            # = sum_u s[u] s[u + L - i]  ... = cross-correlation shifted
            # Build explicitly:
            idx = a - i
            mask = (idx >= 0) & (idx < n)
            v1 = np.zeros(2 * n - 1)
            # (s*delta_i) autoconv: for lag L in [-(n-1), n-1]
            # = sum over k: s[k] * s[k + L - i]  (k+L-i in [0,n-1])
            # This is crosscorr of s with s shifted by i, at lag L.
            # crosscorr(L) = sum_k s[k] s[k + L - i]
            # = sum_k s[k] s[k + (L - i)]
            # = (s s)(L - i)  where (s s)(M) = sum_k s[k] s[k+M] = ac[M + (n-1)]
            # ac is centered: ac[p] = sum_k s[k] s[k + (p - (n-1))]
            # so (s s)(M) = ac[M + (n-1)]
            M = np.arange(-(n - 1), n)  # L - i ranges
            # for L in [-(n-1), n-1], M = L - i
            Mvals = np.arange(-(n - 1), n) - i
            valid = (Mvals >= -(n - 1)) & (Mvals <= (n - 1))
            cc = np.zeros(2 * n - 1)
            cc[valid] = ac[Mvals[valid] + (n - 1)]
            v = 2.0 * s[i] * cc
            new_ac = ac + v
            new_num = float(np.max(np.abs(new_ac)))
            new_den = float(np.sum(s) - 2.0 * s[i]) ** 2
            if new_den <= 0.0:
                continue
            new_score = new_num / new_den
            gain = cur - new_score
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i is None:
            break
        s[best_i] = -s[best_i]
        cur = _score(s)
    return s, cur


def construct():
    best = np.ones(N)
    best_score = _score(best)

    rng = np.random.default_rng(12345)
    # random-sign baselines
    for _ in range(4):
        cand = rng.choice([-1.0, 1.0], size=N)
        sc = _score(cand)
        if sc < best_score:
            best_score = sc
            best = cand.copy()

    # greedy local search from best
    if _left() > 5.0:
        best, best_score = _greedy(best, budget=3.0)

    # a couple more random restarts with greedy if time
    for _ in range(2):
        if _left() < 5.0:
            break
        cand = rng.choice([-1.0, 1.0], size=N)
        sc = _score(cand)
        if sc < best_score:
            best_score = sc
            best = cand.copy()
        if _left() > 5.0:
            cand, sc = _greedy(cand, budget=3.0)
            if sc < best_score:
                best_score = sc
                best = cand

    return _round_to_signed(best)
