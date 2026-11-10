"""
Rewritten program to maximize the cardinality of a pairwise non-coprime subset of [1, 255] containing 255.

Analysis:
255 = 3 * 5 * 17.
To maximize the size of a set S such that gcd(a, b) > 1 for all a, b in S,
we should choose the prime factor p of 255 that yields the largest number of multiples in [1, 255].
The number of multiples of p is floor(255 / p).
- p = 3: floor(255 / 3) = 85
- p = 5: floor(255 / 5) = 51
- p = 17: floor(255 / 17) = 15

The largest set is the multiples of 3.
This set contains 255 (since 255 is divisible by 3).
Any two distinct multiples of 3 have gcd at least 3, so the condition is satisfied.
It can be proven that no larger set exists for n=255.
"""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin

    # 255 = 3 * 5 * 17
    # Precompute gcd compatibility for all pairs in [1,255]
    n_max = 255
    compatible = [[False] * (n_max + 1) for _ in range(n_max + 1)]
    for i in range(1, n_max + 1):
        for j in range(i + 1, n_max + 1):
            c = math.gcd(i, j) > 1
            compatible[i][j] = c
            compatible[j][i] = c

    # Start with all multiples of 3 (largest single-prime clique)
    best_set = set(range(3, n_max + 1, 3))
    best_size = len(best_set)

    # Helper: check if a number is compatible with all in a set
    def compatible_with_all(num, s):
        for x in s:
            if not compatible[num][x]:
                return False
        return True

    # Try to add any number that's compatible with all current members
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for cand in range(1, n_max + 1):
            if cand in best_set:
                continue
            if compatible_with_all(cand, best_set):
                best_set.add(cand)
                best_size += 1
                improved = True
                break

    # Try 1-for-1 swaps: remove one, try to add one that fits
    for _ in range(20):
        if time.time() > deadline:
            break
        improved = False
        for rem in list(best_set):
            if time.time() > deadline:
                break
            # Temporarily remove
            best_set.discard(rem)
            # Try to find a candidate to add
            found = False
            for cand in range(1, n_max + 1):
                if cand in best_set:
                    continue
                if compatible_with_all(cand, best_set):
                    best_set.add(cand)
                    found = True
                    break
            if not found:
                # Revert
                best_set.add(rem)
            # If found, we swapped, size same, but maybe enables further adds
            if found:
                improved = True
                # After swap, try to add more
                sub_improved = True
                while sub_improved:
                    if time.time() > deadline:
                        break
                    sub_improved = False
                    for cand in range(1, n_max + 1):
                        if cand in best_set:
                            continue
                        if compatible_with_all(cand, best_set):
                            best_set.add(cand)
                            best_size += 1
                            sub_improved = True
                            break
        if not improved:
            break

    # Validate
    s_list = sorted(best_set)
    valid = 255 in best_set
    if valid:
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                if math.gcd(s_list[i], s_list[j]) == 1:
                    valid = False
                    break
            if not valid:
                break

    if not valid:
        return list(range(3, n_max + 1, 3))

    return list(best_set)
# EVOLVE-BLOCK-END
