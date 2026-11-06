import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    N = 512
    B = 512

    def lcm(a, b):
        return a // gcd(a, b) * b

    # Precompute compatibility: compat[i] is a bitmask of j such that lcm(i,j) <= B
    compat = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if lcm(i, j) <= B:
                mask |= (1 << j)
        compat[i] = mask

    def elapsed_ok():
        return time.time() < deadline

    def popcount(x):
        return bin(x).count('1')

    def greedy_fill(mask, order=None):
        """Add compatible vertices greedily. If order is None, use 1..N."""
        if order is None:
            order = range(1, N + 1)
        for v in order:
            if mask & (1 << v):
                continue
            if (mask & ~compat[v]) == 0:
                mask |= (1 << v)
        return mask

    def greedy_fill_desc(mask):
        """Add compatible vertices greedily in descending order N..1."""
        for v in range(N, 0, -1):
            if mask & (1 << v):
                continue
            if (mask & ~compat[v]) == 0:
                mask |= (1 << v)
        return mask

    def initial_clique(order):
        mask = 0
        for v in order:
            if (mask & ~compat[v]) == 0:
                mask |= (1 << v)
        mask = greedy_fill(mask)
        return mask

    # Precompute degree
    deg = [0] * (N + 1)
    for i in range(1, N + 1):
        deg[i] = popcount(compat[i])

    # Sort vertices by degree descending
    deg_order = sorted(range(1, N + 1), key=lambda x: deg[x], reverse=True)

    best_mask = 0
    best_size = 0

    # Try several initial orderings
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        deg_order,
        sorted(range(1, N + 1), key=lambda x: deg[x]),
        sorted(range(1, N + 1), key=lambda x: x.bit_length()),
    ]

    for order in orders:
        if not elapsed_ok():
            break
        mask = initial_clique(order)
        size = popcount(mask)
        if size > best_size:
            best_size = size
            best_mask = mask

    # Local search with swaps
    def local_search(mask):
        nonlocal best_size, best_mask
        improved = True
        while improved and elapsed_ok():
            improved = False
            cur_size = popcount(mask)
            clique = [v for v in range(1, N + 1) if mask & (1 << v)]
            
            # Try all single swaps: remove v, add w, then greedy fill
            for i in range(len(clique)):
                if not elapsed_ok():
                    break
                v = clique[i]
                base_mask = mask & ~(1 << v)
                
                # Find all w that can replace v
                candidates = []
                for w in range(1, N + 1):
                    if mask & (1 << w):
                        continue
                    if (base_mask & ~compat[w]) == 0:
                        candidates.append(w)
                
                # Sort by degree descending
                candidates.sort(key=lambda w: deg[w], reverse=True)
                
                for w in candidates[:30]:
                    new_mask = base_mask | (1 << w)
                    new_mask = greedy_fill(new_mask)
                    new_size = popcount(new_mask)
                    if new_size > cur_size:
                        mask = new_mask
                        cur_size = new_size
                        improved = True
                        break
                if improved:
                    break
            
            # Try pure additions
            if not improved:
                new_mask = greedy_fill(mask)
                if popcount(new_mask) > cur_size:
                    mask = new_mask
                    cur_size = popcount(mask)
                    improved = True
            
            if popcount(mask) > best_size:
                best_size = popcount(mask)
                best_mask = mask
        return mask

    local_search(best_mask)

    # Random restarts
    import random
    rng = random.Random(42)
    restarts = 0
    while elapsed_ok() and restarts < 50:
        restarts += 1
        order = list(range(1, N + 1))
        rng.shuffle(order)
        mask = initial_clique(order)
        size = popcount(mask)
        if size >= best_size - 3:
            mask = local_search(mask)
            size = popcount(mask)
            if size > best_size:
                best_size = size
                best_mask = mask

    # Final polish: try 2-swap moves on the best mask
    def two_swap_polish(mask):
        nonlocal best_size, best_mask
        improved = True
        while improved and elapsed_ok():
            improved = False
            cur_size = popcount(mask)
            clique = [v for v in range(1, N + 1) if mask & (1 << v)]
            non_clique = [v for v in range(1, N + 1) if not (mask & (1 << v))]
            
            # Try removing 2 and adding 1 or 2
            for i in range(len(clique)):
                if not elapsed_ok():
                    break
                for j in range(i + 1, len(clique)):
                    v1, v2 = clique[i], clique[j]
                    base_mask = mask & ~(1 << v1) & ~(1 << v2)
                    
                    # Try adding one vertex
                    for w in non_clique[:50]:
                        if (base_mask & ~compat[w]) == 0:
                            new_mask = base_mask | (1 << w)
                            new_mask = greedy_fill(new_mask)
                            if popcount(new_mask) > cur_size:
                                mask = new_mask
                                cur_size = popcount(mask)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            # Try adding 2 vertices
            if not improved:
                for i in range(len(non_clique)):
                    if not elapsed_ok():
                        break
                    for j in range(i + 1, len(non_clique)):
                        w1, w2 = non_clique[i], non_clique[j]
                        if (mask & ~compat[w1]) == 0 and (mask & ~compat[w2]) == 0 and (compat[w1] & (1 << w2)) or (compat[w2] & (1 << w1)):
                            # Check if w1 and w2 are compatible with each other and with mask
                            if (mask & ~compat[w1]) == 0 and (mask & ~compat[w2]) == 0:
                                if lcm(w1, w2) <= B:
                                    new_mask = mask | (1 << w1) | (1 << w2)
                                    new_mask = greedy_fill(new_mask)
                                    if popcount(new_mask) > cur_size:
                                        mask = new_mask
                                        cur_size = popcount(mask)
                                        improved = True
                                        break
                    if improved:
                        break
            
            if popcount(mask) > best_size:
                best_size = popcount(mask)
                best_mask = mask
        return mask

    best_mask = two_swap_polish(best_mask)

    result = [v for v in range(1, N + 1) if best_mask & (1 << v)]
    return result
