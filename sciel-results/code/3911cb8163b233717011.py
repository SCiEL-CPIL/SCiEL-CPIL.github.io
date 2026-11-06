import time
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 75
    n = limit

    # Precompute LCM compatibility matrix
    compat = [[False] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            l = (a * b) // gcd(a, b)
            c = l <= limit
            compat[a][b] = c
            compat[b][a] = c

    # Precompute divisors for each number
    divisors = [set() for _ in range(n + 1)]
    for d in range(1, n + 1):
        for m in range(d, n + 1, d):
            divisors[m].add(d)

    # For a set of maximal elements M, the full set is union of divisors
    def full_set(M):
        S = set()
        for m in M:
            S.update(divisors[m])
        return S

    # Check if x can be added to maximal set M (lcm(x,m) <= limit for all m in M)
    def can_add_maximal(M, x):
        for m in M:
            if not compat[x][m]:
                return False
        return True

    # Local search over maximal-element antichains
    def local_search_maximal(M):
        M = frozenset(M)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            S = full_set(M)
            # 1-for-1 swap: remove one maximal, try adding another
            for e in list(M):
                if time.time() > deadline - 10:
                    break
                M_minus = M - {e}
                for x in range(1, n + 1):
                    if time.time() > deadline - 10:
                        break
                    if x not in M and x != e and can_add_maximal(M_minus, x):
                        M2 = M_minus | {x}
                        S2 = full_set(M2)
                        if len(S2) > len(S):
                            M = M2
                            improved = True
                        break
            # 1-for-2 swap: remove one maximal, try adding two
            for e in list(M):
                if time.time() > deadline - 10:
                    break
                M_minus = M - {e}
                added = 0
                M2 = set(M_minus)
                for x in range(1, n + 1):
                    if time.time() > deadline - 10:
                        break
                    if x not in M2 and x != e and can_add_maximal(M2, x):
                        M2.add(x)
                        added += 1
                        if added >= 2:
                            break
                if len(M2) > len(M_minus):
                    S2 = full_set(M2)
                    S = full_set(M)
                    if len(S2) > len(S):
                        M = frozenset(M2)
                        improved = True
        return full_set(M)

    best = set()

    # Seed strategies: start with various maximal-element sets
    seeds_maximal = []

    # Single highly composite numbers as maximal
    for m in [512, 480, 420, 360, 336, 288, 240, 210, 180, 168, 144, 120, 96, 72, 60, 48, 36, 24, 12, 6, 4, 2]:
        seeds_maximal.append(frozenset([m]))

    # Pairs of compatible numbers
    for m1 in [512, 480, 420, 360, 336, 288, 240]:
        for m2 in [512, 480, 420, 360, 336, 288, 240, 210, 180, 168]:
            if m1 != m2 and compat[m1][m2]:
                seeds_maximal.append(frozenset([m1, m2]))

    # Small ranges as initial maximal sets
    seeds_maximal.append(frozenset(range(1, 17)))
    seeds_maximal.append(frozenset(range(1, 13)))
    seeds_maximal.append(frozenset(range(1, 9)))

    for seed_M in seeds_maximal:
        if time.time() > deadline - 20:
            break
        S = local_search_maximal(seed_M)
        if len(S) > len(best):
            best = S

    # Random restarts with maximal-element seeds
    import random
    rng = random.Random(42)
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        # Random maximal seed: pick a random number, add compatible ones greedily
        M = set()
        start = rng.randint(1, n)
        M.add(start)
        for x in range(1, n + 1):
            if x not in M and can_add_maximal(M, x):
                if rng.random() < 0.3:
                    M.add(x)
        S = local_search_maximal(M)
        if len(S) > len(best):
            best = S

    # Final polish: try to add any remaining compatible elements
    S = set(best)
    changed = True
    while changed:
        changed = False
        for x in range(1, n + 1):
            if time.time() > deadline - 2:
                break
            if x not in S:
                ok = True
                for y in S:
                    if not compat[x][y]:
                        ok = False
                        break
                if ok:
                    S.add(x)
                    changed = True
    best = S

    return sorted(best)
