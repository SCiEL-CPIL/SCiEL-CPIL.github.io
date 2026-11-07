import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    minv = P['minimum_vertices']
    edges = P['edges']
    
    # Build adjacency bitmask array
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
        
    # Localize variables
    adj_local = adj
    n_local = n
    minv_local = minv
    
    best_count = 0
    best_mask = 0
    
    start_time = time.time()
    deadline = start_time + 50.0

    # Recursive backtracking
    # State: idx (current vertex index to consider), mask (selected vertices), deg_mask (bitmask of degrees seen)
    # We iterate vertices from 0 to n-1.
    
    # To allow pruning, we need to know the max possible distinct degrees we can add.
    # Max distinct degrees is n_local.
    
    def dfs(idx, mask, deg_mask):
        nonlocal best_count, best_mask
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning: 
        # 1. If we have selected enough vertices, check if we beat best.
        # Note: We can stop adding vertices if we can't beat best?
        # Actually, adding more vertices might increase distinct degrees.
        # Max possible distinct degrees for a set of size k is k.
        # Current size = mask.bit_count()
        # Remaining vertices = n_local - idx
        # Max size we can reach = current_size + remaining
        # Max distinct degrees we can reach = max(current_distinct, max_size) ?? No.
        # The distinct degrees are determined by the specific subset.
        # A simple bound: current_distinct + remaining_vertices is an upper bound on distinct degrees?
        # No, adding a vertex can only add at most 1 new distinct degree.
        # So max possible distinct = current_distinct.bit_count() + (n_local - idx)
        
        current_distinct = deg_mask.bit_count()
        remaining = n_local - idx
        
        if current_distinct + remaining <= best_count:
            return

        # If we've considered all vertices
        if idx == n_local:
            # Check if valid size
            if mask.bit_count() >= minv_local:
                if current_distinct > best_count:
                    best_count = current_distinct
                    best_mask = mask
            return

        # Option 1: Exclude vertex idx
        dfs(idx + 1, mask, deg_mask)
        
        # Option 2: Include vertex idx
        new_mask = mask | (1 << idx)
        # Calculate degree of idx in new_mask
        deg = (adj_local[idx] & new_mask).bit_count()
        new_deg_mask = deg_mask | (1 << deg)
        
        # Optional pruning: if new_deg_mask.bit_count() + (n_local - idx - 1) <= best_count, skip?
        # This is the same as the check at the top of the next call.
        
        dfs(idx + 1, new_mask, new_deg_mask)

    dfs(0, 0, 0)
    
    # Extract vertices from best_mask
    verts = []
    m = best_mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        verts.append(int(v))
        m ^= lb
        
    return verts
# EVOLVE-BLOCK-END
