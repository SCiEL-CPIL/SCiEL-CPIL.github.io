import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s

    # 255 = 3 * 5 * 17
    # Candidates: x in [1, 255] with gcd(x, 255) > 1
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, 255) > 1:
            candidates.append(x)

    idx_255 = candidates.index(255)
    n = len(candidates)

    # Adjacency matrix as bitmasks
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Greedy initialization: build a large clique starting from 255
    # Strategy: greedily add vertices that are connected to all current clique members
    # Try multiple greedy orders to find a good initial clique
    def greedy_clique(start_idx, order):
        """Greedy clique building. Returns (clique_mask, size)."""
        clique_mask = 1 << start_idx
        size = 1
        available = adj[start_idx]
        for v in order:
            if time.time() > DEADLINE:
                break
            if v == start_idx:
                continue
            if (available & (1 << v)) and (clique_mask & (1 << v)) == 0:
                # Check v is connected to all in clique
                if all((adj[v] & (1 << u)) for u in range(n) if (clique_mask >> u) & 1):
                    clique_mask |= (1 << v)
                    size += 1
                    available &= adj[v]
        return clique_mask, size

    # Generate multiple orderings for greedy
    all_indices = list(range(n))
    best_init_mask = 0
    best_init_size = 0

    # Order 1: by degree descending
    degs = [adj[i].bit_count() for i in range(n)]
    order1 = sorted(all_indices, key=lambda x: -degs[x])
    m, s = greedy_clique(idx_255, order1)
    if s > best_init_size:
        best_init_mask, best_init_size = m, s

    # Order 2: by value descending
    order2 = sorted(all_indices, key=lambda x: -candidates[x])
    m, s = greedy_clique(idx_255, order2)
    if s > best_init_size:
        best_init_mask, best_init_size = m, s

    # Order 3: by value ascending
    order3 = sorted(all_indices, key=lambda x: candidates[x])
    m, s = greedy_clique(idx_255, order3)
    if s > best_init_size:
        best_init_mask, best_init_size = m, s

    # Also try: start with all multiples of 3 that connect to 255
    mult3 = [i for i in range(n) if candidates[i] % 3 == 0]
    mult3_order = sorted(mult3, key=lambda x: -degs[x])
    m, s = greedy_clique(idx_255, mult3_order)
    if s > best_init_size:
        best_init_mask, best_init_size = m, s

    best_size = best_init_size
    best_mask = best_init_mask

    # Bron-Kerbosch with pivot using integer masks
    def bron_kerbosch(clique_mask, candidates_mask, excluded_mask):
        nonlocal best_size, best_mask

        if time.time() > DEADLINE:
            return

        # Pruning: can't beat best
        if candidates_mask.bit_count() + clique_mask.bit_count() <= best_size:
            return

        if candidates_mask == 0:
            if clique_mask.bit_count() > best_size:
                best_size = clique_mask.bit_count()
                best_mask = clique_mask
            return

        # Pivot selection: pick vertex in candidates with max degree in candidates
        pivot = -1
        max_deg = -1
        temp = candidates_mask
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            temp &= temp - 1
            deg = (adj[v] & candidates_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = v

        # Branch on vertices in candidates that are NOT neighbors of pivot
        # (these are the ones that could extend the clique)
        branch_mask = candidates_mask & ~(adj[pivot] if pivot >= 0 else 0)

        temp = branch_mask
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            temp &= temp - 1

            new_clique = clique_mask | (1 << v)
            new_candidates = candidates_mask & adj[v]

            bron_kerbosch(new_clique, new_candidates, 0)

            candidates_mask &= ~(1 << v)

            if time.time() > DEADLINE:
                return

    # Start search: must include 255
    init_clique_mask = 1 << idx_255
    init_candidates = adj[idx_255]

    bron_kerbosch(init_clique_mask, init_candidates, 0)

    # Local search polish
    if time.time() < DEADLINE:
        current_mask = best_mask
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for i in range(n):
                if (current_mask >> i) & 1:
                    continue
                # Check if i is connected to all in current clique
                connected_to_all = True
                temp = current_mask
                while temp:
                    j = (temp & (-temp)).bit_length() - 1
                    temp &= temp - 1
                    if not (adj[i] & (1 << j)):
                        connected_to_all = False
                        break
                if connected_to_all:
                    current_mask |= (1 << i)
                    improved = True
        if current_mask.bit_count() > best_size:
            best_size = current_mask.bit_count()
            best_mask = current_mask

    result = [candidates[i] for i in range(n) if (best_mask >> i) & 1]
    result.sort()

    if 255 not in result:
        result.append(255)

    return result
