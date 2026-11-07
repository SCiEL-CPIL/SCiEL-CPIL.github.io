"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Build adjacency bitmask for fast degree computation
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute popcount for 17-bit integers using a lookup table
    # For 17 bits, we can use int.bit_count() which is very fast in Python 3.10+
    
    def count_distinct_degrees_mask(mask):
        """Count distinct degrees in the induced subgraph using bitmask."""
        degrees = set()
        for v in range(n):
            if mask & (1 << v):
                deg = (adj_mask[v] & mask).bit_count()
                degrees.add(deg)
        return len(degrees)
    
    best_count = 0
    best_mask = 0
    
    # Exhaustive search over all subsets of size >= min_v
    # For n=17, total subsets = 2^17 = 131072, very manageable
    # Use integer bitmask iteration for speed
    for size in range(min_v, n + 1):
        if time.time() > deadline:
            break
        for subset in itertools.combinations(range(n), size):
            if time.time() > deadline:
                break
            mask = 0
            for v in subset:
                mask |= (1 << v)
            distinct = count_distinct_degrees_mask(mask)
            if distinct > best_count:
                best_count = distinct
                best_mask = mask
    
    # Local search: try swapping vertices to improve
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_mask = best_mask
            # Try all possible single swaps
            for add_v in range(n):
                if current_mask & (1 << add_v):
                    continue
                for rem_v in range(n):
                    if not (current_mask & (1 << rem_v)):
                        continue
                    new_mask = (current_mask | (1 << add_v)) & ~(1 << rem_v)
                    distinct = count_distinct_degrees_mask(new_mask)
                    if distinct > best_count:
                        best_count = distinct
                        best_mask = new_mask
                        improved = True
                        break
                if improved:
                    break
        
        # Second phase: try adding vertices (increase size) if beneficial
        if time.time() < deadline:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                current_mask = best_mask
                current_size = current_mask.bit_count()
                if current_size >= n:
                    break
                for add_v in range(n):
                    if current_mask & (1 << add_v):
                        continue
                    new_mask = current_mask | (1 << add_v)
                    distinct = count_distinct_degrees_mask(new_mask)
                    if distinct > best_count:
                        best_count = distinct
                        best_mask = new_mask
                        improved = True
                        break
        
        # Third phase: try removing vertices (decrease size) if beneficial
        if time.time() < deadline:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                current_mask = best_mask
                current_size = current_mask.bit_count()
                if current_size <= min_v:
                    break
                for rem_v in range(n):
                    if not (current_mask & (1 << rem_v)):
                        continue
                    new_mask = current_mask & ~(1 << rem_v)
                    distinct = count_distinct_degrees_mask(new_mask)
                    if distinct > best_count:
                        best_count = distinct
                        best_mask = new_mask
                        improved = True
                        break
        
        # Fourth phase: try 2-swap moves (add 2, remove 2)
        if time.time() < deadline:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                current_mask = best_mask
                current_size = current_mask.bit_count()
                if current_size - 2 < min_v:
                    break
                for add_v in range(n):
                    if current_mask & (1 << add_v):
                        continue
                    for add_v2 in range(n):
                        if add_v2 <= add_v:
                            continue
                        if current_mask & (1 << add_v2):
                            continue
                        for rem_v in range(n):
                            if not (current_mask & (1 << rem_v)):
                                continue
                            for rem_v2 in range(n):
                                if rem_v2 <= rem_v:
                                    continue
                                if not (current_mask & (1 << rem_v2)):
                                    continue
                                new_mask = ((current_mask | (1 << add_v) | (1 << add_v2)) & ~(1 << rem_v) & ~(1 << rem_v2))
                                distinct = count_distinct_degrees_mask(new_mask)
                                if distinct > best_count:
                                    best_count = distinct
                                    best_mask = new_mask
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
    
    # Convert best_mask to list of vertex indices
    result = []
    for v in range(n):
        if best_mask & (1 << v):
            result.append(int(v))
    
    return result
# EVOLVE-BLOCK-END
