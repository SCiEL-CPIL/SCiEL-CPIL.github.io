"""Optimized program: minimize max-correlation via a smooth (low-pass) cosine height profile.

The objective is value = max(np.correlate(h, 1-h, mode='full')) / N * 2, lower is better.
Each lag k contributes sum_i h[i]*(1-h[i+k]) over the valid overlap. If h is a smooth,
slowly-varying signal, then h[i] and h[i+k] are close at every lag, so h[i]*(1-h[i+k])
is small for all i, k — driving down the maximum. A cosine wave is the optimal low-pass
shape (all higher harmonics would only add alignment at some lag). We optimize the period
and amplitude, then polish with coordinate descent, all under an explicit time guard.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 110.0  # safety margin well under the 120 s limit


def _objective(h):
    """The exact evaluator value: max full-correlation of h with (1-h), scaled."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def _cosine_heights(period, amp):
    """Smooth cosine profile with mean 0.5, amplitude amp, period `period` (in samples)."""
    i = np.arange(N)
    return 0.5 + amp * np.cos(2.0 * np.pi * i / period)


def _valid(h):
    """True iff h is finite, in [0,1], and sum within tolerance of N/2."""
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-12) or np.any(h > 1.0 + 1e-12):
        return False
    if abs(np.sum(h) - N / 2.0) > 1e-9 * N:
        return False
    return True


def _clip(h):
    return np.clip(h, 0.0, 1.0)


def construct():
    best_h = _cosine_heights(4.0, 0.5)
    best_v = _objective(best_h)

    # ---- Stage 1: optimize the smooth cosine (period, amplitude) ----
    # Fine grid over period (smaller period = smoother = lower max correlation) and amplitude.
    for amp in (0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1):
        for period in np.arange(2.0, 20.0, 0.25):
            if time.time() > _DEADLINE:
                break
            h = _cosine_heights(period, amp)
            if not _valid(h):
                continue
            v = _objective(h)
            if v < best_v:
                best_v = v
                best_h = h.copy()
        if time.time() > _DEADLINE:
            break

    # ---- Stage 2: coordinate-descent polish (with sum constraint) ----
    # Nudge individual heights to reduce the objective while keeping sum == N/2.
    h = best_h.copy()
    improved = True
    step = 0.02
    while improved and time.time() < _DEADLINE - 10.0:
        improved = False
        cur = _objective(h)
        for i in range(N):
            if time.time() > _DEADLINE - 10.0:
                break
            for delta in (step, -step):
                cand = h.copy()
                cand[i] += delta
                # Rebalance sum by spreading the correction uniformly over the rest.
                if np.any(cand < 0.0) or np.any(cand > 1.0):
                    continue
                diff = np.sum(cand) - N / 2.0
                if abs(diff) > 1e-12:
                    # Apply the correction to a different index to keep cand[i] change minimal.
                    j = (i + 1) % N
                    cand[j] -= diff
                    if cand[j] < 0.0 or cand[j] > 1.0:
                        continue
                if not _valid(cand):
                    continue
                v = _objective(cand)
                if v < cur - 1e-12:
                    h = cand
                    cur = v
                    improved = True
                    break
        # Shrink step once no further improvement is found at the current scale.
        if not improved:
            step *= 0.5
            if step < 1e-4:
                break

    # ---- Final safety: enforce exact validity and return the best valid result ----
    if not _valid(h):
        h = best_h
    if not _valid(h):
        # Fallback to the guaranteed-valid seed.
        h = np.full(N, 0.5)
    return h


if __name__ == "__main__":
    h = construct()
    print("value =", _objective(h))
    print("valid =", _valid(h), "sum =", np.sum(h), "min =", np.min(h), "max =", np.max(h))
