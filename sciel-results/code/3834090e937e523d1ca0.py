"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400
DEADLINE = 145.0

def _compute_ratio(a):
    a = np.asarray(a, dtype=np.float64)
    S = float(np.sum(a))
    if abs(S) < 1e-12:
        return float('inf')
    n = len(a)
    size = 2 * n
    fa = np.fft.rfft(a, size=size)
    conv = np.fft.irfft(fa * np.conj(fa), size=size)
    max_abs = float(np.max(np.abs(conv[:n])))
    return max_abs / (S * S)

def construct():
    start = time.time()
    best_a = np.ones(N)
    best_val = _compute_ratio(best_a)

    # Try structured patterns
    patterns = [np.ones(N)]

    # Mostly +1 with contiguous -1 block
    for num_neg in [1, 2, 4, 8, 16, 32, 64]:
        step = max(1, (N - num_neg) // 30)
        for pos in range(0, N - num_neg + 1, step):
            a = np.ones(N)
            a[pos:pos + num_neg] = -1
            patterns.append(a)

    # Alternating blocks
    for bs in [1, 2, 4, 8, 16, 32, 64]:
        for off in [0, 1]:
            a = np.ones(N)
            for i in range(off, N, 2 * bs):
                a[i:min(i + bs, N)] = -1
            patterns.append(a)

    for a in patterns:
        if time.time() > DEADLINE - 5:
            break
        val = _compute_ratio(a)
        if val < best_val:
            best_val = val
            best_a = a.copy()

    # Local search: flip single elements
    if time.time() < DEADLINE - 10:
        cur = best_a.copy()
        cur_val = best_val
        improved = True
        while improved and time.time() < DEADLINE - 10:
            improved = False
            for i in range(N):
                if time.time() > DEADLINE - 10:
                    break
                cur[i] = -cur[i]
                val = _compute_ratio(cur)
                if val < cur_val:
                    cur_val = val
                    best_a = cur.copy()
                    best_val = val
                    improved = True
                else:
                    cur[i] = -cur[i]

    return best_a
