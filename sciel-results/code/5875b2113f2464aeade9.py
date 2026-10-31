"""Signed equal-width step functions minimizing max|autoconvolution| / (integral)^2.

Objective: score = max_k |c[k]| / (sum_j x[j])^2,  where c = x * x (autoconvolution).
Lower is better. We use signed (+/-1) step heights.

Strategy (structural + local refinement):
  1. Build a structured sign sequence with a dominant DC component (large integral)
     but suppressed side lobes, using a periodic sign pattern biased toward +1.
  2. Refine with a greedy coordinate-descent local search that flips individual
     signs to reduce the objective, with an explicit wall-clock deadline guard.
"""

import time
import numpy as np

N = 400
# Wall-clock budget with a safety margin below the 150 s kill limit.
DEADLINE_S = 135.0


def _autoconv(x):
    """Full autoconvolution c[k] = sum_j x[j] * x[j+k] for k in [0, N-1].
    Uses FFT for O(N log N) cost."""
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    c = np.fft.irfft(f * np.conj(f), n=2 * n)
    return c[:n]


def _score(x):
    """max |autoconvolution| / (integral)^2. Returns inf if integral ~ 0."""
    I = float(np.sum(x))
    if abs(I) < 1e-9:
        return float("inf")
    c = _autoconv(x)
    m = float(np.max(np.abs(c)))
    return m / (I * I)


def _initial_sequence():
    """Structured sign sequence: mostly +1 with periodic negative blocks to
    suppress side lobes while keeping a large positive integral.

    We use x[j] = +1 except on a periodic set of positions where we place -1,
    chosen so the negative positions are evenly spaced (cancels low-lag peaks).
    """
    x = np.ones(N, dtype=np.float64)
    # Place negative signs at every 4th position (a 3:1 positive bias).
    # This keeps integral = N - N/4 = 3N/4 (large) while the regular spacing
    # of negatives suppresses the autoconvolution side lobes.
    x[::4] = -1.0
    return x


def _greedy_local_search(x, deadline):
    """Greedy coordinate descent: repeatedly flip the sign of the index whose
    flip most reduces the objective. Stops on no-improvement sweep or deadline."""
    x = x.copy()
    cur = _score(x)
    if not np.isfinite(cur):
        return x, cur
    n = len(x)
    # Precompute c[0] = sum x^2 = n (constant since x is +/-1).
    # Flipping x[i] -> -x[i] changes the integral by -2*x[i]*I and changes
    # c[k] for all k. We evaluate candidate flips by recomputing the score.
    # To stay within budget, we do full sweeps but early-exit on time.
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        best_i = -1
        best_val = cur
        for i in range(n):
            if time.time() > deadline:
                break
            x[i] = -x[i]
            v = _score(x)
            x[i] = -x[i]  # restore
            if v < best_val - 1e-12:
                best_val = v
                best_i = i
        if best_i >= 0:
            x[best_i] = -x[best_i]
            cur = best_val
            improved = True
    return x, cur


def construct():
    start = time.time()
    deadline = start + DEADLINE_S

    x = _initial_sequence()
    # A few refinement passes, each guarded by the deadline.
    x, cur = _greedy_local_search(x, deadline)

    # Safety: ensure a finite, valid result is always returned.
    if not np.isfinite(cur):
        x = np.ones(N, dtype=np.float64)
        cur = _score(x)

    return x


if __name__ == "__main__":
    res = construct()
    print("score:", _score(res))
