"""Optimized exhaustive search with bitmask operations for distinct degrees."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 87  # Safety margin before 90s kill

    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']

    # Build adjacency as bitmasks for fast operations
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def count_distinct_degrees(mask):
        """Count distinct degrees in induced subgraph given as bitmask."""
        degs = set()
        for v in range(n):
            if mask & (1 << v):
                degs.add((adj_mask[v] & mask).bit_count())
        return len(degs)

    best_count = 0
    best_mask = 0

    # Exhaustive search over all subsets of size min_v to n
    # Total subsets: sum_{k=8}^{17} C(17,k) = 89846
    # Each evaluation is O(n) = 17 operations, so ~1.5M total ops
    # Should complete well within deadline

    for k in range(min_v, n + 1):
        if time.time() > deadline - 1.0:
            break
        for subset in itertools.combinations(range(n), k):
            if time.time() > deadline - 1.0:
                break
            mask = 0
            for v in subset:
                mask |= (1 << v)
            distinct = count_distinct_degrees(mask)
            if distinct > best_count:
                best_count = distinct
                best_mask = mask

    # If we found a good result, also try local search to see if we can improve
    if best_mask != 0 and time.time() < deadline - 5.0:
        # Local search: try adding/removing/swapping vertices
        current_mask = best_mask
        current_count = best_count
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            in_verts = [v for v in range(n) if current_mask & (1 << v)]
            out_verts = [v for v in range(n) if not (current_mask & (1 << v))]

            # Try adding one vertex
            for add_v in out_verts:
                new_mask = current_mask | (1 << add_v)
                new_count = count_distinct_degrees(new_mask)
                if new_count > current_count:
                    current_count = new_count
                    current_mask = new_mask
                    improved = True

            # Try removing one vertex (only if size > min_v)
            if current_mask.bit_count() > min_v:
                for rem_v in in_verts:
                    new_mask = current_mask & ~(1 << rem_v)
                    new_count = count_distinct_degrees(new_mask)
                    if new_count > current_count:
                        current_count = new_count
                        current_mask = new_mask
                        improved = True

            # Try swapping
            for add_v in out_verts:
                for rem_v in in_verts:
                    if time.time() > deadline - 2.0:
                        break
                    new_mask = (current_mask & ~(1 << rem_v)) | (1 << add_v)
                    new_count = count_distinct_degrees(new_mask)
                    if new_count > current_count:
                        current_count = new_count
                        current_mask = new_mask
                        improved = True

        if current_count > best_count:
            best_count = current_count
            best_mask = current_mask

    # Convert best_mask to list of vertices
    if best_mask == 0:
        best_subset = list(range(min_v))
    else:
        best_subset = [v for v in range(n) if best_mask & (1 << v)]

    return [int(v) for v in best_subset]
# EVOLVE-BLOCK-END
