"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def construct():
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']

    # Precompute neighbor masks for O(1) degree queries
    neighbor_masks = [0] * n
    for u, v in edges:
        neighbor_masks[u] |= (1 << v)
        neighbor_masks[v] |= (1 << u)

    best_score = -1
    best_mask = 0
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill

    # Exhaustive search over all subsets with >= min_vertices vertices
    # 2^17 = 131072 total masks; we skip those with < 8 bits
    for mask in range(1 << n):
        # Periodic time check every 4096 masks to reduce overhead
        if (mask & 4095) == 0 and time.time() - start_time > deadline:
            break

        size = mask.bit_count()
        if size < min_vertices:
            continue

        # Compute distinct degrees in the induced subgraph
        deg_set = set()
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = (mask & neighbor_masks[v]).bit_count()
            deg_set.add(deg)
            m -= lsb

        score = len(deg_set)
        if score > best_score:
            best_score = score
            best_mask = mask
            # Upper bound: can't have more distinct degrees than vertices
            if best_score == size:
                # Perfect for this size, but larger subsets could do better
                # Only break if we can't improve
                if best_score >= n:
                    break

    if best_mask == 0:
        # Fallback: first min_vertices vertices
        return [int(i) for i in range(min_vertices)]

    # Extract vertex list from best_mask
    vertices = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        vertices.append(v)
        m -= lsb

    return [int(v) for v in vertices]
# EVOLVE-BLOCK-END
