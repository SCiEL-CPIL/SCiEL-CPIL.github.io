"""Optimized complete search with precomputed degree lookup tables."""
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

    # Precompute adjacency bitmasks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Precompute popcount for all masks up to 2^n
    # This allows O(1) degree lookup via (adj[v] & mask).bit_count()
    # But we can do better: precompute deg[v][mask] for all v and mask
    # Memory: 17 * 131072 = 2,228,224 small ints — acceptable

    full_range = 1 << n
    deg_table = [[0] * full_range for _ in range(n)]

    for v in range(n):
        dv = deg_table[v]
        av = adj[v]
        for mask in range(1, full_range):
            lsb = mask & -mask
            prev = mask ^ lsb
            u = lsb.bit_length() - 1
            if av & (1 << u):
                dv[mask] = dv[prev] + 1
            else:
                dv[mask] = dv[prev]

    best_count = -1
    best_mask = 0
    
    # Precompute bit masks for vertices to speed up mask construction
    bits = [1 << i for i in range(n)]
    dt = deg_table # Local alias for speed

    # Search larger subsets first since they can have more distinct degrees
    for size in range(n, minv - 1, -1):
        if time.time() > deadline:
            break
        # Pruning: if best_count >= size, no subset of this size can improve
        if best_count >= size:
            continue

        for combo in itertools.combinations(range(n), size):
            if time.time() > deadline:
                break

            mask = 0
            for v in combo:
                mask |= bits[v]

            # Count distinct degrees using precomputed table
            degs = set()
            for v in combo:
                degs.add(dt[v][mask])
            c = len(degs)

            if c > best_count:
                best_count = c
                best_mask = mask
                if best_count == size:
                    break

        if best_count == n:
            break

    # Extract vertices from best_mask
    best_vertices = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        best_vertices.append(int(v))
        m ^= lsb

    return best_vertices
# EVOLVE-BLOCK-END
