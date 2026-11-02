import time
import math
import random
import numpy as np
from itertools import combinations

_DEADLINE = None

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    if arr[0] != 0:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    if n < 2:
        return 0.0
    mx = int(arr.max())
    if mx == 0:
        return 0.0

    N = 2 * mx + 1

    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0

    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2

    F = np.fft.rfft(sig, n=fft_size)

    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6)

    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]

    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)

    diffset_size = np.sum(S_diff > 1e-6)

    if sumset_size == 0:
        return 0.0

    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0

    denom = np.log(2 * mx + 1)
    if denom == 0:
        return 0.0

    return 1.0 + np.log(ratio) / denom

# Known optimal Golomb rulers (minimal max element) for n=1..20
_OPTIMAL_GOLOMB = [
    [0],
    [0, 1],
    [0, 1, 3],
    [0, 1, 4, 6],
    [0, 1, 4, 9, 11],
    [0, 1, 4, 10, 12, 17],
    [0, 1, 4, 10, 12, 17, 25],
    [0, 1, 4, 10, 18, 23, 29, 34],
    [0, 1, 4, 10, 18, 23, 29, 37, 44],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72, 85],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72, 85, 106],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72, 85, 106, 129],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72, 85, 106, 129, 153],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72, 85, 106, 129, 153, 179],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72, 85, 106, 129, 153, 179, 208],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72, 85, 106, 129, 153, 179, 208, 238],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72, 85, 106, 129, 153, 179, 208, 238, 275],
    [0, 1, 4, 10, 18, 23, 29, 37, 41, 55, 72, 85, 106, 129, 153, 179, 208, 238, 275, 319],
]

def _local_search(U, max_iter=200, time_limit=5.0):
    """Improve the set U by moving elements (fixed size)."""
    current = sorted(set(U))
    current_score = _score_set(current)
    start_time = time.time()
    n = len(current)
    mx = max(current) if current else 0
    upper = mx + 20

    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        improved = False
        order = list(range(n))
        random.shuffle(order)
        for i in order:
            deltas = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 8, -8, 10, -10, 15, -15]
            random.shuffle(deltas)
            for delta in deltas:
                new_val = current[i] + delta
                if new_val < 0 or new_val > upper:
                    continue
                new_set = current[:]
                new_set[i] = new_val
                new_set = sorted(set(new_set))
                if len(new_set) != n:
                    continue
                new_score = _score_set(new_set)
                if new_score > current_score + 1e-12:
                    current = new_set
                    current_score = new_score
                    mx = max(current)
                    upper = mx + 20
                    improved = True
                    break
            if improved:
                break
    return current, current_score

def _diff_sparse_search(best_set, best_score, deadline):
    """Search 'difference-sparse' AP-subset constructions.

    U is a subset of {0, d, 2d, ..., (k-1)d} of size n. Differences are
    multiples of d spread over a wide range (many distinct), sums are
    multiples of d over a narrower range. We optimize d and the subset.
    """
    global _best_set, _best_score
    rng = random.Random(12345)
    # Try various common differences and sizes
    for n in range(3, 16):
        for d in range(1, 12):
            # candidate positions 0..(n-1)*d, choose n-1 of them (keep 0)
            # positions: 0, d, 2d, ..., (n-1)d  -> we can pick a subsequence
            base = [i * d for i in range(n)]
            # Try random subsets of indices of size n (from a slightly larger pool)
            pool_size = n + 4
            for trial in range(40):
                if time.time() > deadline - 5:
                    return
                idx = [0] + sorted(rng.sample(range(1, pool_size), n - 1))
                U = [i * d for i in idx]
                U = sorted(set(U))
                if 0 not in U:
                    U = [0] + U
                s = _score_set(U)
                if s > _best_score:
                    _best_score = s
                    _best_set = U[:]

def _random_search(best_set, best_score, deadline):
    """Random sets with small max, plus exhaustive for very small sets."""
    global _best_set, _best_score
    rng = random.Random(42)
    # Random sets
    for n in range(3, 14):
        for max_val in range(n, 60):
            for _ in range(300):
                if time.time() > deadline - 5:
                    return
                candidates = list(range(1, max_val + 1))
                if len(candidates) < n - 1:
                    continue
                sample = rng.sample(candidates, n - 1)
                U = sorted([0] + sample)
                s = _score_set(U)
                if s > _best_score:
                    _best_score = s
                    _best_set = U[:]
    # Exhaustive for n=3,4,5 with small max
    for n in range(3, 6):
        for max_val in range(n, 18):
            range_vals = list(range(1, max_val + 1))
            if len(range_vals) < n - 1:
                continue
            for comb in combinations(range_vals, n - 1):
                if time.time() > deadline - 5:
                    return
                U = sorted([0] + list(comb))
                s = _score_set(U)
                if s > _best_score:
                    _best_score = s
                    _best_set = U[:]

def construct():
    global _best_set, _best_score
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin (kill at 150s)

    _best_set = [0, 1]
    _best_score = 0.0

    # 1. Evaluate known Golomb rulers
    for U in _OPTIMAL_GOLOMB:
        if time.time() > deadline - 30:
            break
        if len(U) < 2:
            continue
        s = _score_set(U)
        if s > _best_score:
            _best_score = s
            _best_set = U[:]

    # 2. Difference-sparse AP-subset construction search
    if time.time() < deadline - 25:
        _diff_sparse_search(_best_set, _best_score, deadline)

    # 3. Random + exhaustive small-set search
    if time.time() < deadline - 20:
        _random_search(_best_set, _best_score, deadline)

    # 4. Local search polish on top candidates
    if _best_set is not None and time.time() < deadline - 10:
        improved, improved_score = _local_search(_best_set, max_iter=300, time_limit=15.0)
        if improved_score > _best_score:
            _best_score = improved_score
            _best_set = improved

        # Also polish the best Golomb ruler if it's close
        golomb_best = None
        golomb_best_score = 0.0
        for U in _OPTIMAL_GOLOMB:
            if len(U) < 2:
                continue
            s = _score_set(U)
            if s > golomb_best_score:
                golomb_best_score = s
                golomb_best = U[:]
        if golomb_best is not None and golomb_best_score >= _best_score * 0.99 \
                and golomb_best != _best_set and time.time() < deadline - 8:
            improved2, improved_score2 = _local_search(golomb_best, max_iter=150, time_limit=8.0)
            if improved_score2 > _best_score:
                _best_score = improved_score2
                _best_set = improved2

    if _best_set is None:
        return [0, 1]
    return _best_set

if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:12]}... Size: {len(s)}, Max: {max(s)}, Score: {_score_set(s)}")
