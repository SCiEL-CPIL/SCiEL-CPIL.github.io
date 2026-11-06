import time
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 85
    n = limit

    # Precompute compatibility matrix
    compat = [[False] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            l = (a * b) // gcd(a, b)
            c = l <= limit
            compat[a][b] = c
            compat[b][a] = c

    # Bitsets for divisors of each number
    div_bitsets = [0] * (n + 1)
    for d in range(1, n + 1):
        bit = 1 << (d - 1)
        for m in range(d, n + 1, d):
            div_bitsets[m] |= bit

    # Compatibility bitset: for each x, which y are compatible with x
    compat_bits = [0] * (n + 1)
    for x in range(1, n + 1):
        bits = 0
        for y in range(1, n + 1):
            if compat[x][y]:
                bits |= (1 << (y - 1))
        compat_bits[x] = bits

    # Multiples bitset: for each m, which x are multiples of m
    multiples_bits = [0] * (n + 1)
    for m in range(1, n + 1):
        bits = 0
        for x in range(m, n + 1, m):
            bits |= (1 << (x - 1))
        multiples_bits[m] = bits

    def get_union_bitset(M):
        u = 0
        for m in M:
            u |= div_bitsets[m]
        return u

    def popcount(x):
        return x.bit_count()

    def is_valid_maximal_set(M):
        M_list = list(M)
        for i in range(len(M_list)):
            for j in range(i + 1, len(M_list)):
                if not compat[M_list[i]][M_list[j]]:
                    return False
        # Check no element is a multiple of another
        for i in range(len(M_list)):
            for j in range(len(M_list)):
                if i != j and M_list[i] % M_list[j] == 0:
                    return False
        return True

    def local_search_maximal(M, time_limit):
        M = frozenset(M)
        if not is_valid_maximal_set(M):
            M_list = list(M)
            M_valid = set()
            for m in M_list:
                ok = True
                for o in M_valid:
                    if not compat[m][o] or m % o == 0 or o % m == 0:
                        ok = False
                        break
                if ok:
                    M_valid.add(m)
            M = frozenset(M_valid)

        current_bitset = get_union_bitset(M)
        current_size = popcount(current_bitset)

        improved = True
        while improved and time.time() < time_limit:
            improved = False

            # 1. Try adding a new maximal element
            for x in range(1, n + 1):
                if time.time() > time_limit:
                    break
                if x in M:
                    continue
                # Check compatibility with all in M
                ok = True
                for m in M:
                    if not compat[x][m] or x % m == 0:
                        ok = False
                        break
                if ok:
                    M2 = M | {x}
                    S2 = get_union_bitset(M2)
                    if popcount(S2) > current_size:
                        M = M2
                        current_bitset = S2
                        current_size = popcount(S2)
                        improved = True
                        break

            # 2. 1-for-1 Swap
            for e in list(M):
                if time.time() > time_limit:
                    break
                M_minus = M - {e}
                for x in range(1, n + 1):
                    if time.time() > time_limit:
                        break
                    if x in M and x != e:
                        continue
                    if x == e:
                        continue
                    ok = True
                    for m in M_minus:
                        if not compat[x][m] or x % m == 0:
                            ok = False
                            break
                    if ok:
                        M2 = M_minus | {x}
                        S2 = get_union_bitset(M2)
                        if popcount(S2) > current_size:
                            M = M2
                            current_bitset = S2
                            current_size = popcount(S2)
                            improved = True
                            break

            # 3. 1-for-2 Swap (Remove 1, Add 2)
            for e in list(M):
                if time.time() > time_limit:
                    break
                M_minus = M - {e}
                best_pair_size = -1
                best_pair = None
                for x in range(1, n + 1):
                    if time.time() > time_limit:
                        break
                    if x in M or x == e:
                        continue
                    ok_x = True
                    for m in M_minus:
                        if not compat[x][m] or x % m == 0:
                            ok_x = False
                            break
                    if not ok_x:
                        continue
                    for y in range(x + 1, n + 1):
                        if time.time() > time_limit:
                            break
                        if y in M or y == e:
                            continue
                        if not compat[x][y]:
                            continue
                        if y % x == 0 or x % y == 0:
                            continue
                        ok_y = True
                        for m in M_minus:
                            if not compat[y][m] or y % m == 0:
                                ok_y = False
                                break
                        if not ok_y:
                            continue
                        M2 = M_minus | {x, y}
                        S2 = get_union_bitset(M2)
                        size2 = popcount(S2)
                        if size2 > best_pair_size:
                            best_pair_size = size2
                            best_pair = (x, y)
                if best_pair and best_pair_size > current_size:
                    M = M_minus | set(best_pair)
                    current_bitset = get_union_bitset(M)
                    current_size = best_pair_size
                    improved = True

            # 4. 2-for-2 Swap (Remove 2, Add 2)
            M_list = list(M)
            for i in range(len(M_list)):
                if time.time() > time_limit:
                    break
                for j in range(i + 1, len(M_list)):
                    if time.time() > time_limit:
                        break
                    e1, e2 = M_list[i], M_list[j]
                    M_minus = M - {e1, e2}
                    best_pair_size = -1
                    best_pair = None
                    for x in range(1, n + 1):
                        if time.time() > time_limit:
                            break
                        if x in M:
                            continue
                        ok_x = True
                        for m in M_minus:
                            if not compat[x][m] or x % m == 0:
                                ok_x = False
                                break
                        if not ok_x:
                            continue
                        for y in range(x + 1, n + 1):
                            if time.time() > time_limit:
                                break
                            if y in M:
                                continue
                            if not compat[x][y]:
                                continue
                            if y % x == 0 or x % y == 0:
                                continue
                            ok_y = True
                            for m in M_minus:
                                if not compat[y][m] or y % m == 0:
                                    ok_y = False
                                    break
                            if not ok_y:
                                continue
                            M2 = M_minus | {x, y}
                            S2 = get_union_bitset(M2)
                            size2 = popcount(S2)
                            if size2 > best_pair_size:
                                best_pair_size = size2
                                best_pair = (x, y)
                    if best_pair and best_pair_size > current_size:
                        M = M_minus | set(best_pair)
                        current_bitset = get_union_bitset(M)
                        current_size = best_pair_size
                        improved = True

        return get_union_bitset(M)

    best_bitset = 0
    best_size = 0

    # Seeds: single high numbers
    seeds_maximal = []
    for m in [512, 480, 420, 360, 336, 288, 240, 210, 180, 168, 144, 120, 96, 72, 60, 48, 36, 24, 12, 6, 4, 2]:
        if m <= limit:
            seeds_maximal.append(frozenset({m}))

    # Seeds: pairs of compatible numbers
    for a in range(2, n + 1):
        for b in range(a + 1, n + 1):
            if compat[a][b] and b % a != 0 and a % b != 0:
                seeds_maximal.append(frozenset({a, b}))
                if len(seeds_maximal) > 300:
                    break
        if len(seeds_maximal) > 300:
            break

    # Seeds: ranges
    for upper in [4, 6, 8, 12, 16, 24, 32, 48, 64]:
        seeds_maximal.append(frozenset(range(1, upper + 1)))

    # Seeds: divisors of highly composite numbers (as maximal elements)
    # Maximal elements are those not dividing any other in the set
    for base in [504, 420, 360, 336, 240, 180, 120, 96, 72, 48, 36, 24]:
        divs = [d for d in range(1, n + 1) if base % d == 0 and d <= n]
        # Maximal elements: those not dividing another divisor
        maximal = set()
        for d in divs:
            if not any(d != e and e % d == 0 for e in divs):
                maximal.add(d)
        if maximal:
            seeds_maximal.append(frozenset(maximal))

    # Also try: all numbers whose lcm with each other is <= 512
    # Greedy: pick numbers in order of decreasing "usefulness"
    for seed in seeds_maximal:
        if time.time() > deadline - 20:
            break
        S_bitset = local_search_maximal(seed, deadline - 20)
        S_size = popcount(S_bitset)
        if S_size > best_size:
            best_size = S_size
            best_bitset = S_bitset

    # Random restarts
    import random
    rng = random.Random(42)
    for _ in range(300):
        if time.time() > deadline - 5:
            break
        M = set()
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        for x in candidates:
            ok = True
            for m in M:
                if not compat[x][m] or x % m == 0:
                    ok = False
                    break
            if ok:
                M.add(x)
        if len(M) > 0:
            S_bitset = local_search_maximal(frozenset(M), deadline - 5)
            S_size = popcount(S_bitset)
            if S_size > best_size:
                best_size = S_size
                best_bitset = S_bitset

    # Convert bitset to list
    best = []
    for i in range(1, n + 1):
        if best_bitset & (1 << (i - 1)):
            best.append(i)

    # Final polish: try adding any compatible number
    changed = True
    while changed:
        changed = False
        for x in range(1, n + 1):
            if x not in best and all(compat[x][y] for y in best):
                best.append(x)
                best_bitset |= (1 << (x - 1))
                changed = True

    return sorted(best)
