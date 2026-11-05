"""Closed-form optimal solution for the third autocorrelation inequality task.

Analysis:
  value = abs(2 * N * max_k (h*h)[k] / (sum h)^2)

  The autoconvolution's lag-0 entry is sum(h_i^2) > 0.
  For any lag k != 0, |sum_i h_i h_{i+k}| <= sqrt(sum h_i^2 * sum h_{i+k}^2)
  <= sum h_i^2  (Cauchy-Schwarz, since the shifted sum is a sub-sum).
  Hence max_k (h*h)[k] = sum(h_i^2) (the positive lag-0 value is the largest).

  Therefore value = 2 * N * sum(h_i^2) / (sum h_i)^2.

  By Cauchy-Schwarz, (sum h_i)^2 <= N * sum(h_i^2), so
  sum(h_i^2)/(sum h_i)^2 >= 1/N, giving value >= 2*N*(1/N) = 2.0.

  Equality holds iff all h_i are equal. Thus value >= 2.0 for every feasible
  h, and the constant vector attains exactly 2.0. The seed score is 2.0/2.0 = 1.0,
  which is therefore the global maximum achievable score. No perturbation can
  improve it; any deviation strictly increases value and lowers the score.

So the optimal program simply returns the closed-form optimal constant vector.
"""

import time
import numpy as np

N = 400
_DEADLINE = 150.0  # evaluator kills after 150 s wall-clock; keep a margin.


def construct():
    t0 = time.time()
    # Closed-form global optimum: any non-zero constant vector.
    # np.ones is exact in floating point and yields value = 2.0, score = 1.0.
    h = np.ones(N, dtype=float)

    # Safety: verify finite, non-zero sum, and within deadline.
    if not np.all(np.isfinite(h)) or np.sum(h) == 0.0:
        # Fallback (never expected to trigger for ones).
        h = np.ones(N, dtype=float)

    if time.time() - t0 > _DEADLINE - 5.0:
        # Should never happen for this O(N) construction.
        return h
    return h
