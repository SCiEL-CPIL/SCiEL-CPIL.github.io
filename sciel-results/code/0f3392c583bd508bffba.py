import time
import random
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 75
    n = limit

    # Precompute compatibility as bitmasks
    # compat_mask[i] has bit j set if lcm(i, j) <= n
    # Note: bit 0 is unused, we use bits 1..n
    compat_mask = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, n + 1):
            if i * j // gcd(i, j) <= n:
                mask |= (1 << j)
        compat_mask[i] = mask

    # Helper: check if x can be added to S (represented as bitmask)
    # S is a bitmask. x is an integer.
    # x is compatible with S if compat_mask[x] covers all bits in S.
    # i.e., S & ~compat_mask[x] == 0
    def can_add(S_mask, x):
        return (S_mask & ~compat_mask[x]) == 0

    def complete(S_mask):
        # Greedily add all possible elements
        # Note: If x is not compatible with S, it will not become compatible
        # as S grows. So a single pass is sufficient to reach a maximal set.
        for x in range(1, n + 1):
            if time.time() > deadline - 5:
                break
            if (S_mask & (1 << x)) == 0 and can_add(S_mask, x):
                S_mask |= (1 << x)
        return S_mask

    def mask_to_set(S_mask):
        return [i for i in range(1, n + 1) if (S_mask & (1 << i))]

    def local_search(S_mask):
        S_mask = complete(S_mask)
        current_size = bin(S_mask).count('1')
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            S_list = mask_to_set(S_mask)
            
            # Move 1: Remove 1 element, then complete.
            # This is stronger than 1-for-1 or 1-for-2 as it adds ALL possible elements.
            for e in S_list:
                if time.time() > deadline - 10:
                    break
                S_minus = S_mask & ~(1 << e)
                S2 = complete(S_minus)
                new_size = bin(S2).count('1')
                if new_size > current_size:
                    S_mask = S2
                    current_size = new_size
                    improved = True
                    S_list = mask_to_set(S_mask) # Update list for next iteration
            
            # Move 2: Remove 2 elements, then complete.
            # Only attempt if we have time and previous move didn't succeed immediately
            if not improved and time.time() < deadline - 15:
                for i in range(len(S_list)):
                    if time.time() > deadline - 15:
                        break
                    for j in range(i + 1, len(S_list)):
                        if time.time() > deadline - 15:
                            break
                        e1 = S_list[i]
                        e2 = S_list[j]
                        S_minus = S_mask & ~(1 << e1) & ~(1 << e2)
                        S2 = complete(S_minus)
                        new_size = bin(S2).count('1')
                        if new_size > current_size:
                            S_mask = S2
                            current_size = new_size
                            improved = True
                            break
                    if improved:
                        break

        return S_mask

    best_mask = 0
    best_size = 0

    # Strategy 1: Greedy from small numbers
    S = 0
    for x in range(1, n + 1):
        if can_add(S, x):
            S |= (1 << x)
    S = local_search(S)
    if bin(S).count('1') > best_size:
        best_size = bin(S).count('1')
        best_mask = S

    # Strategy 2: Greedy by compatibility count (most compatible first)
    comp_counts = [bin(compat_mask[i]).count('1') for i in range(n + 1)]
    order = sorted(range(1, n + 1), key=lambda x: -comp_counts[x])
    S = 0
    for x in order:
        if time.time() > deadline - 30:
            break
        if can_add(S, x):
            S |= (1 << x)
    S = local_search(S)
    if bin(S).count('1') > best_size:
        best_size = bin(S).count('1')
        best_mask = S

    # Strategy 3: Random Restarts
    rng = random.Random(42)
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        S = 0
        # Random start
        start_size = rng.randint(1, 10)
        for _ in range(start_size):
            x = rng.randint(1, n)
            S |= (1 << x)
        S = local_search(S)
        if bin(S).count('1') > best_size:
            best_size = bin(S).count('1')
            best_mask = S

    # Strategy 4: Specific seed based on analysis (1..23 + multiples)
    S = 0
    for x in range(1, 24):
        S |= (1 << x)
    # Try to add multiples of 23, 22, etc.
    for x in range(24, n + 1):
        if can_add(S, x):
            S |= (1 << x)
    S = local_search(S)
    if bin(S).count('1') > best_size:
        best_size = bin(S).count('1')
        best_mask = S

    # Strategy 5: Divisors of highly composite numbers
    for m in [360, 420, 480, 504, 252, 336]:
        if time.time() > deadline - 10:
            break
        S = 0
        for x in range(1, n + 1):
            if m % x == 0:
                S |= (1 << x)
        S = local_search(S)
        if bin(S).count('1') > best_size:
            best_size = bin(S).count('1')
            best_mask = S

    return mask_to_set(best_mask)
