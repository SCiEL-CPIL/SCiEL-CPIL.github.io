import time
import numpy as np

N = 400
START_TIME = time.time()
DEADLINE = START_TIME + 135.0

def _c3_prime_val(h):
    """
    Compute max_k |autocorr_k| / (sum h)^2 for signed step function h of length N.
    """
    h = np.asarray(h, dtype=float)
    total = h.sum()
    if abs(total) < 1e-12:
        return np.inf
    denom = total * total
    ac = np.correlate(h, h, mode='full')
    return np.max(np.abs(ac)) / denom

def construct():
    # The theoretical minimum for the metric max|ac| / (sum h)^2 for any real sequence
    # is 1/N, achieved by a constant sequence (e.g., all ones).
    # This is derived from ac[0] = sum(h^2) >= (sum(h))^2 / N.
    # Thus max|ac| >= ac[0] >= (sum(h))^2 / N.
    # Ratio >= 1/N.
    # The seed program likely achieves this, resulting in score 1.0.
    # We return the optimal constant sequence directly to ensure speed and correctness.
    return np.ones(N)
