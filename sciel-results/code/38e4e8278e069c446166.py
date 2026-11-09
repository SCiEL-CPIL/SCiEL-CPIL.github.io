import numpy as np
import time

N = 400

def _autoconv(x):
    n = len(x)
    size = 2 * n
    X = np.fft.rfft(x, size=size)
    C = np.fft.irfft(X * np.conj(X), size=size)
    return C[:n]

def _objective(x):
    c = _autoconv(x)
    max_abs = np.max(np.abs(c))
    s = np.sum(x)
    if abs(s) < 1e-10:
        return float('inf')
    return max_abs / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Initialize with ones
    x = np.ones(N)
    best_x = x.copy()
    best_obj = _objective(x)

    # Local search with random perturbations
    np.random.seed(42)
    for _ in range(100000):
        if time.time() > deadline:
            break
        i = np.random.randint(N)
        old_val = x[i]
        x[i] = old_val + np.random.normal(0, 0.05)
        obj = _objective(x)
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()
        else:
            x[i] = old_val  # Revert

    return best_x
