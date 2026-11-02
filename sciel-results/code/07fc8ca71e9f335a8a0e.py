"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_max_abs(x):
    """Compute max absolute value of linear autocorrelation."""
    n = len(x)
    max_val = 0.0
    for k in range(n):
        ac = np.dot(x[:n-k], x[k:])
        abs_ac = abs(ac)
        if abs_ac > max_val:
            max_val = abs_ac
    return max_val


def _try_flip(x, i):
    """Try flipping bit i and return new max abs autocorr."""
    x[i] = -x[i]
    val = _autocorr_max_abs(x)
    x[i] = -x[i]  # undo
    return val


def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # 10s safety margin
    
    # Start with all ones
    x = np.ones(N, dtype=np.int8)
    best_val = _autocorr_max_abs(x)
    best_x = x.copy()
    
    # Local search: try flipping each bit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            old = x[i]
            x[i] = -old
            val = _autocorr_max_abs(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
                improved = True
            else:
                x[i] = old
    
    return best_x
# EVOLVE-BLOCK-END
