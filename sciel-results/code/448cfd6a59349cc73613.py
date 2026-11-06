import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 512
    limit = N

    # Precompute compatibility using bitsets for speed
    comp_bits = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(i, N + 1):
            g = gcd(i, j)
            lcm_val = (i // g) * j
            if lcm_val <= limit:
                mask |= (1 << j)
        comp_bits[i] = mask
        if i != i:  # never true, but for symmetry
            pass
    
    # Also build reverse for quick lookup
    comp_set = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            lcm_val = (i // g) * j
            if lcm_val <= limit:
                comp_set[i].add(j)
                if i != j:
                    comp_set[j].add(i)

    def is_valid_add(S, x):
        cx = comp_set[x]
        for y in S:
            if y not in cx:
                return False
        return True

    def complete(S):
        S = set(S)
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if time.time() > deadline - 2:
                    return S
                if x not in S and is_valid_add(S, x):
                    S.add(x)
                    changed = True
        return S

    def local_search(S):
        S = set(S)
        S = complete(S)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
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
            # 2-for-2 swaps (limited)
            if time.time() < deadline - 15:
                elements = list(S)
                for i in range(len(elements)):
                    if time.time() > deadline - 15:
                        break
                    e1 = elements[i]
                    for j in range(i + 1, len(elements)):
                        if time.time() > deadline - 15:
                            break
                        e2 = elements[j]
                        S_minus = S - {e1, e2}
                        S2 = set(S_minus)
                        added = 0
                        for x in range(1, N + 1):
                            if time.time() > deadline - 15:
                                break
                            if x not in S2 and is_valid_add(S2, x):
                                S2.add(x)
                                added += 1
                                if added >= 2:
                                    break
                        if len(S2) > len(S):
                            S = complete(S2)
                            improved = True
            # Random kicks
            if time.time() < deadline - 20 and not improved:
                for _ in range(5):
                    if time.time() > deadline - 20:
                        break
                    if len(S) < 2:
                        break
                    e1, e2 = random.sample(list(S), 2)
                    S_minus = S - {e1, e2}
                    S2 = complete(S_minus)
                    if len(S2) > len(S):
                        S = S2
                        improved = True

        return S

    # Build good initial solutions based on structure
    # Powers of 2 are always mutually compatible
    # Odd numbers: check compatibility
    
    # Strategy: start with all powers of 2, then greedily add odd numbers
    best = set()
    best_size = 0

    # Initial: powers of 2 up to 512
    p2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    S = set(p2)
    S = complete(S)
    S = local_search(S)
    if len(S) > best_size:
        best = S
        best_size = len(S)

    # Try: small numbers 1..k
    for k in [15, 18, 20, 22, 24, 26]:
        if time.time() > deadline - 30:
            break
        S = set(range(1, k + 1))
        S = complete(S)
        S = local_search(S)
        if len(S) > best_size:
            best = S
            best_size = len(S)

    # Random restarts
    rng = random.Random(42)
    for it in range(100):
        if time.time() > deadline - 10:
            break
        # Random initial set
        seed_size = rng.randint(5, 20)
        seed = set(rng.sample(range(1, N + 1), seed_size))
        S = complete(seed)
        S = local_search(S)
        if len(S) > best_size:
            best = S
            best_size = len(S)
            # Kick
            if time.time() < deadline - 10 and len(best) > 2:
                for _ in range(3):
                    if time.time() > deadline - 10:
                        break
                    e = rng.choice(sorted(best))
                    S2 = local_search(best - {e})
                    if len(S2) > best_size:
                        best = S2
                        best_size = len(S2)

    return sorted(best)
