"""Optimized low-discrepancy signs on arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256
_DEADLINE = 85.0  # safety margin below 90s kill

def _build_progressions(n=_N):
    """Return list of numpy index arrays for all APs in [1,n] (0-indexed)."""
    progs = []
    for step in range(1, n + 1):
        for start in range(0, n, step):
            idx = np.arange(start, n, step, dtype=np.int32)
            progs.append(idx)
    return progs

def _max_disc(signs, progs):
    """Max |signed sum| over all APs."""
    best = 0
    for idx in progs:
        s = abs(int(np.sum(signs[idx])))
        if s > best:
            best = s
    return best

def _thue_morse(n):
    """Thue-Morse ±1 sequence (low-discrepancy base)."""
    signs = np.ones(n, dtype=np.int8)
    for i in range(1, n):
        # parity of popcount(i)
        c = 0
        v = i
        while v:
            c ^= (v & 1)
            v >>= 1
        signs[i] = 1 if c == 0 else -1
    return signs

def construct():
    t0 = time.time()
    rng = random.Random(42)

    # Precompute all APs
    progs = _build_progressions(_N)
    n_progs = len(progs)

    # Start from Thue-Morse (good low-discrepancy structure)
    signs = _thue_morse(_N).astype(np.int8)
    cur_best = _max_disc(signs, progs)

    best_signs = signs.copy()
    best_val = cur_best

    # Simulated annealing
    T = 8.0
    T_min = 0.05
    cooling = 0.997

    while time.time() - t0 < _DEADLINE:
        if T < T_min:
            # Reheat occasionally if stuck
            T = 2.0
            # Random restart with perturbation
            for _ in range(rng.randint(5, 20)):
                i = rng.randrange(_N)
                signs[i] *= -1
            cur_best = _max_disc(signs, progs)
            continue

        # Flip a random position
        i = rng.randrange(_N)
        old = signs[i]
        signs[i] = -old

        # Compute new max discrepancy
        new_best = 0
        for idx in progs:
            s = abs(int(np.sum(signs[idx])))
            if s > new_best:
                new_best = s
                if new_best > cur_best + 1:
                    break  # early exit: clearly worse

        delta = new_best - cur_best
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            cur_best = new_best
            if cur_best < best_val:
                best_val = cur_best
                best_signs = signs.copy()
        else:
            signs[i] = old

        T *= cooling

    # Final: try to improve best_signs with local search
    if time.time() - t0 < _DEADLINE - 5:
        s = best_signs.copy()
        v = best_val
        improved = True
        while improved and time.time() - t0 < _DEADLINE - 2:
            improved = False
            for i in range(_N):
                s[i] *= -1
                nv = _max_disc(s, progs)
                if nv < v:
                    v = nv
                    improved = True
                else:
                    s[i] *= -1
            if v < best_val:
                best_val = v
                best_signs = s.copy()

    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
