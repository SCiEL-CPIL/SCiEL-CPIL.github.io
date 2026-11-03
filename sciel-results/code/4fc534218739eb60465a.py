import time
from math import log
import numpy as np

def _fast_score(U):
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    n = len(U)
    # Sumset via autocorrelation of indicator vector (linear, non-cyclic)
    m = 2 * max_u + 1
    N = 1
    while N < 2 * m - 1:
        N <<= 1
    f = np.zeros(N, dtype=np.float64)
    f[U] = 1.0
    F = np.fft.rfft(f)
    auto = np.fft.irfft(F * np.conj(F))
    sumset_size = int(np.count_nonzero(auto[:m] > 0.5))
    # Difference set via cyclic autocorrelation (size 2*max_u+1)
    N2 = 1
    while N2 < 2 * m:
        N2 <<= 1
    F2 = np.fft.rfft(f, n=N2)
    diff_auto = np.fft.irfft(F2 * np.conj(F2))
    diffset_size = int(np.count_nonzero(diff_auto[:m] > 0.5))
    ratio = diffset_size / sumset_size
    if ratio <= 1.0:
        return 1.0
    return 1.0 + log(ratio) / log(2 * max_u + 1)

def compute_score(U):
    try:
        return _fast_score(U)
    except Exception:
        # Fallback to exact O(n^2) computation
        U = sorted(set(U))
        if 0 not in U:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
        if ratio <= 1:
            return 1.0
        return 1.0 + log(ratio) / log(2 * max_u + 1)

def construct():
    start = time.time()
    deadline = start + 110  # 40s safety margin under 150s kill

    def time_ok():
        return time.time() < deadline

    best_U = [0, 1, 3]
    best_score = compute_score(best_U)

    def consider(U):
        nonlocal best_U, best_score
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
        return s

    # --- Structured families: large-base powers and b-1 families ---
    # For {0, b, b^2, ...}: ratio -> 2, exponent -> 1 + ln2/ln(2b^k)
    # For {0, b-1, (b-1)^2, ...}: similar
    for b in range(2, 2000):
        if not time_ok():
            break
        # powers of b
        for k in range(2, 12):
            if not time_ok():
                break
            U = [b**i for i in range(k)]
            consider(U)
        # (b-1) family
        for k in range(2, 12):
            if not time_ok():
                break
            U = [(b - 1) ** i for i in range(k)]
            consider(U)

    # Small structured seeds
    seeds = [[0, 1, 3], [0, 1, 3, 7], [0, 1, 3, 9], [0, 1, 3, 7, 15]]
    for s in seeds:
        if not time_ok():
            break
        consider(s)

    # --- Greedy local search with rich candidate pool ---
    def greedy_grow(U):
        U = sorted(set(U))
        cur = compute_score(U)
        improved = True
        while improved and time_ok():
            improved = False
            mx = max(U)
            # Candidate pool: all integers up to a bound + structured large jumps
            bound = min(8000, mx * 3 + 200)
            cands = list(range(1, bound + 1))
            # Structured large jumps (to exploit large-base regime)
            for mult in (2, 3, 4, 5, 7, 10, 16, 32):
                cands.append(mx * mult)
            cands += [mx * mx, mx * mx * 2, mx * 3 + 1]
            cands = sorted(set(c for c in cands if c > 0))
            best_cand = None
            best_cand_score = cur
            for c in cands:
                if not time_ok():
                    break
                if c in U:
                    continue
                new_U = U + [c]
                ns = compute_score(new_U)
                if ns > best_cand_score:
                    best_cand_score = ns
                    best_cand = c
            if best_cand is not None:
                U = U + [best_cand]
                U = sorted(set(U))
                cur = best_cand_score
                improved = True
        return U, cur

    for seed in seeds:
        if not time_ok():
            break
        U, s = greedy_grow(seed)
        if s > best_score:
            best_score = s
            best_U = list(U)

    # Final: try greedy grow from the current best as well
    if time_ok():
        U, s = greedy_grow(best_U)
        if s > best_score:
            best_score = s
            best_U = list(U)

    return best_U
