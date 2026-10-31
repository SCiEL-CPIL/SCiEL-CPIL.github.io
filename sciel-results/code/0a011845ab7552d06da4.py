import numpy as np
import time

N = 400

def _compute_autoconv(x):
    n = len(x)
    C = np.zeros(n, dtype=np.int32)
    for k in range(n):
        C[k] = np.dot(x[:n-k], x[k:])
    return C

def _thue_morse(n):
    tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    return 1 - 2 * tm

def construct():
    start_time = time.time()
    deadline = 140.0

    x = _thue_morse(N).astype(np.int8)
    C = _compute_autoconv(x)
    max_abs = int(np.max(np.abs(C)))

    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for i in range(N):
            if (time.time() - start_time) > deadline:
                break
            v = x[i]
            delta_C = np.zeros(N, dtype=np.int32)

            max_k1 = N - 1 - i
            if max_k1 >= 1:
                delta_C[1:max_k1 + 1] += -2 * v * x[i + 1:i + max_k1 + 1]

            max_k2 = i
            if max_k2 >= 1:
                idx = i - np.arange(1, max_k2 + 1)
                delta_C[1:max_k2 + 1] += -2 * v * x[idx]

            new_max_abs = int(np.max(np.abs(C + delta_C)))
            if new_max_abs < max_abs:
                x[i] = -v
                C += delta_C
                max_abs = new_max_abs
                improved = True

    return x
