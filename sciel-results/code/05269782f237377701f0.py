# EVOLVE-BLOCK-START
import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.time() + 620.0
    margin = 5.0

    # 1. Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # 2. Precompute conflict masks using bit operations
    conflict_mask = [0] * m
    for v in range(m):
        mask = 0
        for u in range(m):
            if v != u and ((v - u) % m) in forbidden:
                mask |= (1 << u)
        conflict_mask[v] = mask

    remove_mask = [conflict_mask[v] | (1 << v) for v in range(m)]

    # 3. Fast popcount using int.bit_count()
    def popcount(x):
        return x.bit_count()

    best_set = []
    best_size = 0
    full_mask = (1 << m) - 1

    # Node counter for time-based pruning
    node_count = [0]

    # Precompute bit positions for faster iteration
    # We iterate over set bits using lsb trick

    def search(avail, current_set, size):
        nonlocal best_set, best_size

        node_count[0] += 1
        if time.time() > deadline - margin:
            return

        # Pruning: upper bound
        if size + popcount(avail) <= best_size:
            return

        if avail == 0:
            if size > best_size:
                best_size = size
                best_set = list(current_set)
            return

        # Select vertex with max degree in induced subgraph
        best_v = -1
        max_deg = -1
        temp_avail = avail
        while temp_avail:
            lsb = temp_avail & -temp_avail
            v = lsb.bit_length() - 1
            temp_avail ^= lsb
            deg = popcount(avail & conflict_mask[v])
            if deg > max_deg:
                max_deg = deg
                best_v = v

        if best_v == -1:
            return

        v = best_v

        # Branch 1: Include v
        new_avail = avail & ~remove_mask[v]
        current_set.append(v)
        search(new_avail, current_set, size + 1)
        current_set.pop()

        # Branch 2: Exclude v
        new_avail = avail & ~(1 << v)
        search(new_avail, current_set, size)

    def greedy_init():
        avail = full_mask
        s = []
        while avail:
            min_v = -1
            min_deg = m + 1
            temp = avail
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = popcount(avail & conflict_mask[v])
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            if min_v == -1:
                break
            s.append(min_v)
            avail &= ~remove_mask[min_v]
        return s

    # Multiple greedy restarts with different orderings
    def greedy_with_order(order):
        avail = full_mask
        s = []
        # Try vertices in the given order, skipping those already removed
        for v in order:
            if (avail >> v) & 1:
                s.append(v)
                avail &= ~remove_mask[v]
        return s

    # Generate multiple orderings
    import random
    rng = random.Random(42)

    # Standard greedy
    init_set = greedy_init()
    if len(init_set) > best_size:
        best_size = len(init_set)
        best_set = list(init_set)

    # Try different random orderings
    for _ in range(20):
        order = list(range(m))
        rng.shuffle(order)
        s = greedy_with_order(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = list(s)

    # Run B&B with time limit
    search(full_mask, [], 0)

    res = sorted(best_set)

    # Verify validity
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if (res[i] - res[j]) % m in forbidden:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: use greedy result
        res = sorted(init_set)

    return np.array(res, dtype=np.int64)

# EVOLVE-BLOCK-END
