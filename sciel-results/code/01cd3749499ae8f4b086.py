"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    def elapsed():
        return time.time() - start_time

    # Generate powerful numbers up to L
    # Every powerful number n = prod p_i^{e_i}, e_i >= 2
    # Can be written as a^2 * b^3 (generates all powerful numbers)
    L = 100_000_000  # 10^8

    powerful_set = set()
    powerful_set.add(1)

    max_b = int(round(L ** (1.0/3.0))) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerful_set.add(n)

    powerful = sorted(powerful_set)
    n_p = len(powerful)

    # Full O(n^2) enumeration: count all pair sums
    # This is the strategy that scored 4.8 in inspiration 1
    count = defaultdict(int)
    for i in range(n_p):
        a = powerful[i]
        for j in range(i, n_p):
            b = powerful[j]
            N = a + b
            if N > 10**18:
                break
            count[N] += 1
        if elapsed() > deadline - 30.0:
            break

    if count:
        best_N = max(count, key=count.get)
        best_count = count[best_N]
        pset = set(powerful)
        pairs = []
        for a in powerful:
            b = best_N - a
            if b >= a and b in pset:
                pairs.append([a, b])
        return {'N': best_N, 'pairs': pairs}

    # Fallback: find at least one valid pair
    return {'N': 2, 'pairs': [[1, 1]]}
# EVOLVE-BLOCK-END
