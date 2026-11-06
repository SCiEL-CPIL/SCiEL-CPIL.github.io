"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

def generate_powerful_numbers(limit):
    """Generate all powerful numbers <= limit.
    A powerful number has every prime factor appearing with exponent >= 2.
    Every such n can be written as a^2 * b^3 (b squarefree), so iterating over
    all b and all a and collecting a^2*b^3 covers every powerful number.
    Duplicates are handled by the set."""
    if limit < 1:
        return []
    powerfuls = set()
    max_b = int(limit ** (1/3)) + 2
    for b in range(1, max_b + 1):
        b3 = b * b * b
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= limit:
                powerfuls.add(n)
    return sorted(powerfuls)

def count_pairs_for_N(N, P, Pset):
    """Return list of pairs [a, b] with a <= b, a+b=N, a,b in P (a,b powerful)."""
    pairs = []
    for a in P:
        if a > N // 2:
            break
        b = N - a
        if b in Pset:
            pairs.append([a, b])
    return pairs

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin vs 150s wall-clock limit

    # ---- Phase 1: find candidate Ns with NumPy vectorized pairwise sums ----
    # Pure-Python double loop is O(n^2) and too slow; vectorize with bincount.
    # n1 ~ count of powerfuls up to L1. For L1=10^9, n1 ~ 3e4, n1^2 ~ 9e8 sums.
    L1 = 10**9
    pow_list_1 = generate_powerful_numbers(L1)
    n1 = len(pow_list_1)

    P1 = np.array(pow_list_1, dtype=np.int64)
    i_idx, j_idx = np.triu_indices(n1)
    sums = P1[i_idx] + P1[j_idx]
    max_sum = int(sums.max()) if n1 else 0
    counts = np.bincount(sums, minlength=max_sum + 1)
    # Exclude index 0 (sum 0 never occurs, but keeps argmax safe)
    if counts.size > 0:
        counts[0] = 0

    # Top candidates by count, tie-break by smaller N
    topk = min(300, counts.size)
    order = np.argsort(-counts, kind='stable')
    top_counts = counts[order[:topk]]
    # among ties keep the smaller N (stable sort already keeps ascending N within equal counts)
    candidates = [int(x) for x in order[:topk] if counts[x] > 0]

    # ---- Phase 2: verify/extend against progressively larger limits ----
    best_N = 0
    best_pairs = []

    def evaluate_limit(L):
        nonlocal best_N, best_pairs
        P = generate_powerful_numbers(L)
        if not P:
            return
        Pset = set(P)
        maxp = P[-1]
        for N in candidates:
            if N > 2 * maxp:
                continue
            pairs = count_pairs_for_N(N, P, Pset)
            if len(pairs) > len(best_pairs):
                best_N = N
                best_pairs = pairs

    # Start with a limit large enough to already validate the Phase-1 pairs
    evaluate_limit(10**10)

    # Keep pushing the limit higher while time permits; each larger limit can
    # reveal additional powerful-summand pairs for the same N.
    for L2 in (10**11, 10**12, 10**13):
        if time.time() > deadline - 25:
            break
        evaluate_limit(L2)

    # ---- Fallback ----
    if not best_pairs:
        # Ensure a valid answer: 1 is powerful, so 2 = 1 + 1.
        return {'N': 2, 'pairs': [[1, 1]]}

    # Final safety: guarantee 2 <= N <= 10^18
    if best_N < 2 or best_N > 10**18:
        return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
