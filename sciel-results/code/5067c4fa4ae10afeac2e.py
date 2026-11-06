import time
from math import gcd
import random

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 85
    n = limit

    compat = [[False] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            l = (a * b) // gcd(a, b)
            c = l <= limit
            compat[a][b] = c
            compat[b][a] = c

    # Bitsets for divisors
    div_bitsets = [0] * (n + 1)
    for d in range(1, n + 1):
        bit = 1 << (d - 1)
        for m in range(d, n + 1, d):
            div_bitsets[m] |= bit

    def get_union_bitset(M):
        u = 0
        for m in M:
            u |= div_bitsets[m]
        return u

    def popcount(x):
        return bin(x).count('1')

    def is_valid_maximal_set(M):
        M_list = list(M)
        for i in range(len(M_list)):
            for j in range(i + 1, len(M_list)):
                if not compat[M_list[i]][M_list[j]]:
                    return False
        return True

    def local_search_maximal(M, time_limit):
        M = frozenset(M)
        if not is_valid_maximal_set(M):
            M_list = list(M)
            M_valid = set()
            for m in M_list:
                if all(compat[m][o] for o in M_valid):
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
                if x not in M:
                    if all(compat[x][m] for m in M) and not any(x % m == 0 for m in M):
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
                    if x not in M and x != e:
                        if all(compat[x][m] for m in M_minus) and not any(x % m == 0 for m in M_minus):
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
                    if not all(compat[x][m] for m in M_minus) or any(x % m == 0 for m in M_minus):
                        continue
                    for y in range(x + 1, n + 1):
                        if time.time() > time_limit:
                            break
                        if y in M or y == e:
                            continue
                        if not compat[x][y]:
                            continue
                        if not all(compat[y][m] for m in M_minus) or any(y % m == 0 for m in M_minus):
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
    
    seeds_maximal = []
    for m in [512, 480, 420, 360, 336, 288, 240, 210, 180, 168, 144, 120, 96, 72, 60, 48, 36, 24, 12, 6, 4, 2]:
        if m <= limit:
            seeds_maximal.append(frozenset({m}))

    for a in range(2, n + 1):
        for b in range(a + 1, n + 1):
            if compat[a][b]:
                seeds_maximal.append(frozenset({a, b}))
                if len(seeds_maximal) > 200:
                    break
        if len(seeds_maximal) > 200:
            break

    for upper in [4, 6, 8, 12, 16, 24, 32]:
        seeds_maximal.append(frozenset(range(1, upper + 1)))

    for seed in seeds_maximal:
        if time.time() > deadline - 20:
            break
        S_bitset = local_search_maximal(seed, deadline - 20)
        S_size = popcount(S_bitset)
        if S_size > best_size:
            best_size = S_size
            best_bitset = S_bitset

    rng = random.Random(42)
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        M = set()
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        for x in candidates:
            if all(compat[x][m] for m in M) and not any(x % m == 0 for m in M):
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
            
    # Final polish
    for x in range(1, n + 1):
        if x not in best and all(compat[x][y] for y in best):
            best.append(x)
            best_bitset |= (1 << (x - 1))

    return sorted(best)
