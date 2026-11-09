"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    N = 128
    nums = list(range(1, N + 1))

    # Precompute gcd matrix
    g = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            g[i][j] = gcd(i, j)
            g[j][i] = g[i][j]

    def is_bad_triple(a, b, c):
        return g[a][b] == g[a][c] == g[b][c]

    def can_add(x, S):
        # check if adding x to S creates a bad triple
        for a in S:
            for b in S:
                if a < b and is_bad_triple(a, b, x):
                    return False
        return True

    def valid_set(S):
        for a, b, c in combinations(S, 3):
            if is_bad_triple(a, b, c):
                return False
        return True

    # ---- Greedy with multiple orderings ----
    import random
    rng = random.Random(12345)

    best = set()

    # Ordering 1: natural
    for order in [nums, list(reversed(nums)), sorted(nums, key=lambda x: (-x.bit_count(), x))]:
        S = []
        for x in order:
            if can_add(x, S):
                S.append(x)
        if len(S) > len(best):
            best = set(S)

    # Random orderings
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        order = nums[:]
        rng.shuffle(order)
        S = []
        for x in order:
            if can_add(x, S):
                S.append(x)
        if len(S) > len(best):
            best = set(S)

    # ---- Local search: add / swap improvements ----
    S = list(best)
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Try adding any missing element
        for x in nums:
            if x not in S and can_add(x, S):
                S.append(x)
                improved = True
        # Try swap: remove one, add one (or two)
        # For each pair (remove r, add a) check validity
        for r in list(S):
            S_no_r = [y for y in S if y != r]
            for a in nums:
                if a in S:
                    continue
                if can_add(a, S_no_r):
                    # check full validity
                    cand = S_no_r + [a]
                    if valid_set(cand) and len(cand) > len(S):
                        S = cand
                        improved = True
                        break
            if improved:
                break
            # also try adding two by removing one
            for a in nums:
                if a in S:
                    continue
                for b in nums:
                    if b in S or b == a:
                        continue
                    cand = S_no_r + [a, b]
                    if valid_set(cand) and len(cand) > len(S):
                        S = cand
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if len(S) > len(best):
            best = set(S)

    # ---- Final greedy polish on best ----
    S = list(best)
    for x in nums:
        if x not in S and can_add(x, S):
            S.append(x)
    if len(S) > len(best):
        best = set(S)

    # Ensure nonempty
    if not best:
        best = {1}

    return list(best)
# EVOLVE-BLOCK-END
