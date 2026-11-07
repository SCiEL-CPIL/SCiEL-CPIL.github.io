import numpy as np
import time

N = 100
HALF = N // 2


def _value(h):
    corr = np.correlate(h, 1.0 - h, mode='full')
    return float(corr.max()) / N * 2.0


def _local_search(h, deadline, rng):
    ones = [i for i in range(N) if h[i] == 1]
    zeros = [i for i in range(N) if h[i] == 0]
    best = h.copy()
    best_val = _value(best)
    no_improve = 0
    while time.time() < deadline:
        # escape local minima with a random kick
        if no_improve > 2000:
            for _ in range(5):
                i = ones[rng.integers(len(ones))]
                j = zeros[rng.integers(len(zeros))]
                if i == j:
                    continue
                h[i], h[j] = h[j], h[i]
                ones[ones.index(i)] = j
                zeros[zeros.index(j)] = i
            no_improve = 0
        i = ones[rng.integers(len(ones))]
        j = zeros[rng.integers(len(zeros))]
        if i == j:
            continue
        # perform swap
        h[i], h[j] = h[j], h[i]
        ones[ones.index(i)] = j
        zeros[zeros.index(j)] = i
        val = _value(h)
        if val < best_val - 1e-12:
            best_val = val
            best = h.copy()
            no_improve = 0
        else:
            # revert
            h[i], h[j] = h[j], h[i]
            ones[ones.index(j)] = i
            zeros[zeros.index(i)] = j
            no_improve += 1
    return best, best_val


def construct():
    deadline = time.time() + 140.0  # 10s safety margin under the 150s kill
    # seed fallback (valid: sum=50, values in [0,1])
    best = np.full(N, 0.5)
    best_val = _value(best)
    base_rng = np.random.default_rng(1000)
    restart = 0
    # a few structured seeds to give the search a head start
    starts = []
    alt = np.zeros(N); alt[0::2] = 1.0; starts.append(alt.copy())
    block = np.zeros(N); block[:HALF] = 1.0; starts.append(block.copy())
    while time.time() < deadline - 5.0:
        if restart < len(starts):
            h = starts[restart].copy()
        else:
            h = np.zeros(N)
            idx = base_rng.permutation(N)
            h[idx[:HALF]] = 1.0
        rng = np.random.default_rng(1000 + restart)
        slice_deadline = min(deadline - 5.0, time.time() + 2.0)
        h2, v2 = _local_search(h, slice_deadline, rng)
        if v2 < best_val:
            best_val = v2
            best = h2.copy()
        restart += 1
    return best
