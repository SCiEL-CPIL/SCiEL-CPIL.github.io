"""Optimized search for maximum distinct degrees in induced subgraph."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    edges = P['edges']
    minv = P['minimum_vertices']

    # Precompute adjacency bitmasks and neighbor lists
    adj = [0] * n
    neighbors = [[] for _ in range(n)]
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
        neighbors[a].append(b)
        neighbors[b].append(a)

    def count_distinct_degrees(mask):
        """Count distinct degrees of vertices in the induced subgraph defined by mask."""
        degs = set()
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            d = (adj[v] & mask).bit_count()
            degs.add(d)
            m ^= lsb
        return len(degs)

    def get_vertices(mask):
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(int(v))
            m ^= lsb
        return verts

    best_count = -1
    best_mask = 0

    # Phase 1: Exhaustive search over size 8 (the minimum)
    # This is the most important since it's the constraint boundary
    for combo in itertools.combinations(range(n), 8):
        if time.time() > deadline:
            break
        mask = 0
        for v in combo:
            mask |= (1 << v)
        c = count_distinct_degrees(mask)
        if c > best_count:
            best_count = c
            best_mask = mask
            if best_count == 8:
                break  # Can't do better than 8 for size 8

    # Phase 2: Try to extend the best size-8 solution to larger sizes
    # Greedy extension: try adding one vertex at a time
    if time.time() < deadline:
        current_mask = best_mask
        current_count = best_count
        
        # Try all sizes from 9 to 17
        for target_size in range(9, n + 1):
            if time.time() > deadline:
                break
            
            # Try to extend by adding vertices that increase distinct degrees
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try adding each vertex not in current_mask
                for v in range(n):
                    if current_mask & (1 << v):
                        continue
                    new_mask = current_mask | (1 << v)
                    c = count_distinct_degrees(new_mask)
                    if c > current_count:
                        current_mask = new_mask
                        current_count = c
                        improved = True
                        if c > best_count:
                            best_count = c
                            best_mask = new_mask
                        break  # Restart with new mask
                if current_mask.bit_count() >= target_size:
                    break
            
            # Also try local search: swap vertices
            if time.time() < deadline:
                for _ in range(50):  # Limited swaps
                    if time.time() > deadline:
                        break
                    found_improvement = False
                    # Try swapping one vertex in with one out
                    in_verts = get_vertices(current_mask)
                    out_verts = [v for v in range(n) if not (current_mask & (1 << v))]
                    for vi in in_verts:
                        for vo in out_verts:
                            new_mask = current_mask ^ (1 << vi) ^ (1 << vo)
                            c = count_distinct_degrees(new_mask)
                            if c > current_count:
                                current_mask = new_mask
                                current_count = c
                                found_improvement = True
                                if c > best_count:
                                    best_count = c
                                    best_mask = new_mask
                                break
                        if found_improvement:
                            break
                    if not found_improvement:
                        break

    # Phase 3: Exhaustive search over larger sizes if time permits
    # Only search sizes where we could potentially beat best_count
    for size in range(9, n + 1):
        if time.time() > deadline:
            break
        if best_count >= size:
            continue  # Can't beat current best
        
        # Sample some combinations rather than exhaustive
        import random
        random.seed(42)
        all_combos = list(itertools.combinations(range(n), size))
        # Shuffle and try a subset
        random.shuffle(all_combos)
        max_samples = min(len(all_combos), 5000)
        for combo in all_combos[:max_samples]:
            if time.time() > deadline:
                break
            mask = 0
            for v in combo:
                mask |= (1 << v)
            c = count_distinct_degrees(mask)
            if c > best_count:
                best_count = c
                best_mask = mask
                if best_count == size:
                    break

    # Extract vertices from best_mask
    return get_vertices(best_mask)
# EVOLVE-BLOCK-END
