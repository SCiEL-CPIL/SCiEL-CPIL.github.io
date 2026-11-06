import time
import random
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 75
    n = limit

    # Precompute compatibility as bitmasks
    compat_mask = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, n + 1):
            if i * j // gcd(i, j) <= n:
                mask |= (1 << j)
        compat_mask[i] = mask

    def can_add(S_mask, x):
        return (S_mask & ~compat_mask[x]) == 0

    def complete(S_mask):
        # Greedily add all possible elements in a single pass.
        # Since adding elements only restricts future additions, one pass suffices for maximality.
        for x in range(1, n + 1):
            if time.time() > deadline - 5:
                break
            if (S_mask & (1 << x)) == 0 and can_add(S_mask, x):
                S_mask |= (1 << x)
        return S_mask

    def mask_to_set(S_mask):
        return [i for i in range(1, n + 1) if (S_mask & (1 << i))]

    def popcount(x):
        return bin(x).count('1')

    def local_search(S_mask):
        S_mask = complete(S_mask)
        current_size = popcount(S_mask)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            S_list = mask_to_set(S_mask)
            size = len(S_list)
            
            # Move 1: Remove 1 element, then complete.
            for idx, e in enumerate(S_list):
                if time.time() > deadline - 10:
                    break
                S_minus = S_mask & ~(1 << e)
                S2 = complete(S_minus)
                new_size = popcount(S2)
                if new_size > current_size:
                    S_mask = S2
                    current_size = new_size
                    improved = True
                    S_list = mask_to_set(S_mask)
                    size = len(S_list)
                    # Restart scan from beginning as list changed
                    idx = -1 
                    break
            if improved:
                continue # Restart the while loop

            # Move 2: Remove 2 elements, then complete.
            if time.time() < deadline - 15:
                found_improvement = False
                for i in range(size):
                    if time.time() > deadline - 15:
                        break
                    e1 = S_list[i]
                    for j in range(i + 1, size):
                        if time.time() > deadline - 15:
                            break
                        e2 = S_list[j]
                        S_minus = S_mask & ~(1 << e1) & ~(1 << e2)
                        S2 = complete(S_minus)
                        new_size = popcount(S2)
                        if new_size > current_size:
                            S_mask = S2
                            current_size = new_size
                            improved = True
                            found_improvement = True
                            break
                    if found_improvement:
                        break
            if improved:
                continue

            # Move 3: Remove 3 elements, then complete. (Heavier, for deeper local optima)
            if time.time() < deadline - 20:
                found_improvement = False
                # Limit the search space for 3-removals to keep it tractable
                # Only try removing triples that are "close" in value or index? 
                # Or just sample? Let's try a limited systematic search first.
                # To save time, we only try removing triples where at least two are "small" (<= 30) 
                # or just sample random triples from the current set if it's large.
                
                # Actually, let's just do a limited systematic search on the first 40 elements of the sorted list
                # and random samples.
                
                # Systematic on small indices
                limit_idx = min(size, 40)
                for i in range(limit_idx):
                    if time.time() > deadline - 20:
                        break
                    e1 = S_list[i]
                    for j in range(i + 1, limit_idx):
                        if time.time() > deadline - 20:
                            break
                        e2 = S_list[j]
                        for k in range(j + 1, limit_idx):
                            if time.time() > deadline - 20:
                                break
                            e3 = S_list[k]
                            S_minus = S_mask & ~(1 << e1) & ~(1 << e2) & ~(1 << e3)
                            S2 = complete(S_minus)
                            new_size = popcount(S2)
                            if new_size > current_size:
                                S_mask = S2
                                current_size = new_size
                                improved = True
                                found_improvement = True
                                break
                        if found_improvement:
                            break
                    if found_improvement:
                        break
                
                # Random sampling of triples if systematic didn't work
                if not found_improvement and time.time() < deadline - 10:
                    rng_local = random.Random(time.time_ns())
                    for _ in range(200):
                        if time.time() > deadline - 10:
                            break
                        if size < 3:
                            break
                        indices = rng_local.sample(range(size), 3)
                        e1, e2, e3 = S_list[indices[0]], S_list[indices[1]], S_list[indices[2]]
                        S_minus = S_mask & ~(1 << e1) & ~(1 << e2) & ~(1 << e3)
                        S2 = complete(S_minus)
                        new_size = popcount(S2)
                        if new_size > current_size:
                            S_mask = S2
                            current_size = new_size
                            improved = True
                            found_improvement = True
                            break
            if improved:
                continue

        return S_mask

    best_mask = 0
    best_size = 0

    def update_best(S):
        nonlocal best_mask, best_size
        s = popcount(S)
        if s > best_size:
            best_size = s
            best_mask = S

    # Strategy 1: Greedy from small numbers
    S = 0
    for x in range(1, n + 1):
        if can_add(S, x):
            S |= (1 << x)
    S = local_search(S)
    update_best(S)

    # Strategy 2: Greedy by compatibility count (most compatible first)
    comp_counts = [popcount(compat_mask[i]) for i in range(n + 1)]
    order = sorted(range(1, n + 1), key=lambda x: -comp_counts[x])
    S = 0
    for x in order:
        if time.time() > deadline - 30:
            break
        if can_add(S, x):
            S |= (1 << x)
    S = local_search(S)
    update_best(S)

    # Strategy 3: Random Restarts
    rng = random.Random(42)
    for _ in range(300):
        if time.time() > deadline - 10:
            break
        S = 0
        # Random start
        start_size = rng.randint(1, 15)
        for _ in range(start_size):
            x = rng.randint(1, n)
            S |= (1 << x)
        S = local_search(S)
        update_best(S)

    # Strategy 4: Specific seed based on analysis (1..23 + multiples)
    S = 0
    for x in range(1, 24):
        S |= (1 << x)
    for x in range(24, n + 1):
        if can_add(S, x):
            S |= (1 << x)
    S = local_search(S)
    update_best(S)

    # Strategy 5: Divisors of highly composite numbers
    for m in [360, 420, 480, 504, 252, 336, 432, 288, 384, 448]:
        if time.time() > deadline - 10:
            break
        S = 0
        for x in range(1, n + 1):
            if m % x == 0:
                S |= (1 << x)
        S = local_search(S)
        update_best(S)
        
        # Also try mixing divisors of m with small numbers
        S2 = S
        for x in range(1, 30):
            if can_add(S2, x):
                S2 |= (1 << x)
        S2 = local_search(S2)
        update_best(S2)

    # Strategy 6: Greedy with randomized tie-breaking
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        S = 0
        # Sort by compatibility, but shuffle ties
        items = []
        for x in range(1, n + 1):
            items.append((comp_counts[x], rng.random(), x))
        items.sort(key=lambda t: (-t[0], t[1]))
        for _, _, x in items:
            if can_add(S, x):
                S |= (1 << x)
        S = local_search(S)
        update_best(S)

    return mask_to_set(best_mask)
