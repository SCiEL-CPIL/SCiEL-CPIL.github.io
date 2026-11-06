import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 90s limit, 5s margin
    
    N = 512
    limit = N

    # Precompute compatibility sets for all numbers 1..N
    # comp[i] is a set of j such that lcm(i, j) <= limit
    comp = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        ci = comp[i]
        for j in range(i, N + 1):
            g = gcd(i, j)
            lcm_val = (i // g) * j
            if lcm_val <= limit:
                ci.add(j)
                if i != j:
                    comp[j].add(i)

    def is_valid_add(S, x):
        cx = comp[x]
        for y in S:
            if y not in cx:
                return False
        return True

    def complete(S):
        S = set(S)
        changed = True
        while changed:
            changed = False
            # Iterate in increasing order for determinism and speed (smaller numbers usually have more compatibility)
            for x in range(1, N + 1):
                if time.time() > deadline - 2:
                    return S
                if x not in S and is_valid_add(S, x):
                    S.add(x)
                    changed = True
        return S

    def local_search(S, max_iters=50):
        S = set(S)
        S = complete(S)
        improved = True
        iters = 0
        while improved and time.time() < deadline - 5 and iters < max_iters:
            improved = False
            iters += 1
            # 1-for-1 swaps
            for e in list(S):
                if time.time() > deadline - 5:
                    break
                S_minus = S - {e}
                for x in range(1, N + 1):
                    if time.time() > deadline - 5:
                        break
                    if x != e and x not in S and is_valid_add(S_minus, x):
                        S2 = complete(S_minus | {x})
                        if len(S2) > len(S):
                            S = S2
                            improved = True
                        break
            # 1-for-2 swaps
            for e in list(S):
                if time.time() > deadline - 5:
                    break
                S_minus = S - {e}
                added = 0
                S2 = set(S_minus)
                for x in range(1, N + 1):
                    if time.time() > deadline - 5:
                        break
                    if x not in S2 and is_valid_add(S2, x):
                        S2.add(x)
                        added += 1
                        if added >= 2:
                            break
                if len(S2) > len(S):
                    S = complete(S2)
                    improved = True
            # 2-for-2 swaps
            if time.time() < deadline - 10:
                elements = list(S)
                for i in range(len(elements)):
                    if time.time() > deadline - 10:
                        break
                    e1 = elements[i]
                    for j in range(i + 1, len(elements)):
                        if time.time() > deadline - 10:
                            break
                        e2 = elements[j]
                        S_minus = S - {e1, e2}
                        S2 = set(S_minus)
                        added = 0
                        for x in range(1, N + 1):
                            if time.time() > deadline - 10:
                                break
                            if x not in S2 and is_valid_add(S2, x):
                                S2.add(x)
                                added += 1
                                if added >= 2:
                                    break
                        if len(S2) > len(S):
                            S = complete(S2)
                            improved = True
                            
        return S

    best = set()
    best_size = 0
    rng = random.Random(12345)

    # Strategy 1: Fixed Power of 2 construction
    # For each k, find max subset of odd numbers m in [1, L] with lcm(m1, m2) <= L
    # Then S = { 2^k * m | m in M }
    for k in range(10):
        if time.time() > deadline - 20:
            break
        pow2 = 1 << k
        L = N // pow2
        # Find max independent set of odd numbers in [1, L] with limit L
        # This is a smaller instance.
        odd_nums = [m for m in range(1, L + 1) if m % 2 == 1]
        # Greedy construction for this sub-problem
        M = set()
        for m in odd_nums:
            # Check compatibility with all in M
            ok = True
            for y in M:
                g = gcd(m, y)
                if (m // g) * y > L:
                    ok = False
                    break
            if ok:
                M.add(m)
        
        # Build S from M
        S = set()
        for m in M:
            S.add(pow2 * m)
        
        S = complete(S)
        S = local_search(S)
        
        if len(S) > best_size:
            best = S
            best_size = len(S)

    # Strategy 2: Standard greedy and local search
    S = set(range(1, 24))
    S = complete(S)
    S = local_search(S)
    if len(S) > best_size:
        best = S
        best_size = len(S)

    # Strategy 3: Random restarts
    for it in range(100):
        if time.time() > deadline - 10:
            break
        seed = {1}
        order = list(range(2, N + 1))
        rng.shuffle(order)
        for x in order:
            if time.time() > deadline - 10:
                break
            if is_valid_add(seed, x):
                seed.add(x)
        S = local_search(seed)
        if len(S) > best_size:
            best = S
            best_size = len(S)
            if time.time() < deadline - 10 and len(best) > 1:
                e = rng.choice(sorted(best))
                S2 = local_search(best - {e})
                if len(S2) > best_size:
                    best = S2
                    best_size = len(S2)

    return sorted(best)
