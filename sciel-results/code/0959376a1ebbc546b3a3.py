"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_size = P['minimum_vertices']
    
    # Build adjacency as bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    best_subset = list(range(min_size))
    best_count = 0
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Precompute full degree for each vertex in the full graph
    full_deg = [adj_mask[i].bit_count() for i in range(n)]
    
    # Branch and bound search over all subsets of size >= min_size
    # State: (vertex_index, current_mask, current_degrees_list, current_size)
    # We process vertices 0..n-1, deciding to include or exclude each.
    
    # To enable pruning, we track:
    # - current_mask: bitmask of included vertices so far
    # - deg_changes: for each included vertex, we need to compute its degree in the final subset
    #   But degrees change as we add more vertices. So we can't easily track running degrees.
    # 
    # Alternative approach: 
    # Since n=17 is small, enumerate all 2^17 subsets but with pruning.
    # A subset is valid if it has at least min_size vertices.
    # 
    # Pruning strategy:
    # When we have a partial assignment (first k vertices decided), 
    # let included = set of included vertices among 0..k-1
    # let remaining = n - k vertices not yet decided
    # 
    # The final subset will have size = len(included) + x, where 0 <= x <= remaining
    # We need len(included) + x >= min_size, so x >= min_size - len(included)
    # 
    # For the distinct degrees: in the final subset, each vertex v's degree is 
    # the number of its neighbors in the final subset.
    # 
    # This is hard to bound tightly during partial assignment.
    # 
    # Better approach: since 2^17 = 131072, just enumerate all subsets directly
    # but compute degrees efficiently.
    
    # Direct enumeration of all 2^n subsets
    for mask in range(1 << n):
        if time.time() > deadline:
            break
        
        size = mask.bit_count()
        if size < min_size:
            continue
        
        # Upper bound on distinct degrees: at most 'size' distinct values
        # If best_count >= size, we can't improve
        if best_count >= size:
            continue
        
        # Compute degrees for each vertex in the subset
        deg_set = set()
        for v in range(n):
            if mask & (1 << v):
                d = (adj_mask[v] & mask).bit_count()
                deg_set.add(d)
        
        cnt = len(deg_set)
        if cnt > best_count:
            best_count = cnt
            best_subset = [v for v in range(n) if mask & (1 << v)]
    
    return [int(v) for v in best_subset]
# EVOLVE-BLOCK-END
