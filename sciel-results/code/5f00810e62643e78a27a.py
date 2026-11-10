import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix as bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute clique and independent set status for all masks
    # mask is an integer 0 to 2^n - 1
    size = 1 << n
    is_clique = [False] * size
    is_ind = [False] * size
    is_valid = [False] * size
    
    is_clique[0] = True
    is_ind[0] = True
    is_valid[0] = True
    
    for mask in range(1, size):
        # Get lowest set bit
        lb = mask & -mask
        v = lb.bit_length() - 1
        rest = mask ^ lb
        
        # Check clique: v must be connected to all in rest, and rest must be clique
        if is_clique[rest]:
            # Check if v is connected to all vertices in rest
            # adj[v] has bits for neighbors
            # rest has bits for vertices in rest
            # If (rest & ~adj[v]) == 0, then v is connected to all in rest
            if (rest & ~adj[v]) == 0:
                is_clique[mask] = True
                
        # Check independent set: v must NOT be connected to any in rest, and rest must be independent
        if is_ind[rest]:
            # If (rest & adj[v]) == 0, then v is not connected to any in rest
            if (rest & adj[v]) == 0:
                is_ind[mask] = True
                
        is_valid[mask] = is_clique[mask] or is_ind[mask]

    # Greedy initial solution to have a bound
    def greedy_partition():
        used = [False] * n
        parts = []
        for v in range(n):
            if used[v]:
                continue
            # Try to find a valid block containing v, preferably large
            # Simple: just start a new part with v, then try to add others?
            # Let's just do the simple seed-like greedy first for a bound
            part = [v]
            used[v] = True
            # Try to extend
            for u in range(n):
                if used[u]:
                    continue
                trial = part + [u]
                # Check if trial is clique or ind set
                # Convert to mask for fast check
                m = 0
                for x in trial:
                    m |= (1 << x)
                if is_valid[m]:
                    part.append(u)
                    used[u] = True
            parts.append(part)
        return parts

    best_parts = greedy_partition()
    best_count = len(best_parts)
    
    start_time = time.time()
    deadline = 50.0 # 50 seconds
    
    # DFS to find optimal
    # State: remaining_mask, current_parts_list
    # We always pick the smallest vertex in remaining_mask to start the next block
    
    def dfs(remaining, current_parts):
        nonlocal best_parts, best_count
        
        if time.time() - start_time > deadline:
            return True # Signal timeout
        
        if remaining == 0:
            if len(current_parts) < best_count:
                best_count = len(current_parts)
                best_parts = [list(p) for p in current_parts]
            return False
        
        # Pruning: if we already have >= best_count parts, no point continuing
        # Note: we add 1 part for the next block, so if len(current_parts) + 1 >= best_count, prune?
        # Actually, if len(current_parts) >= best_count, we can't improve.
        if len(current_parts) >= best_count:
            return False
            
        # Pick the lowest set bit in remaining
        lb = remaining & -remaining
        v = lb.bit_length() - 1
        
        # Iterate through all submasks of 'remaining' that contain 'v'
        # Let sub = remaining. We want sub such that (sub & lb) != 0 and sub is a subset of remaining.
        # Standard submask iteration:
        # sub = remaining
        # while sub:
        #   if (sub & lb) and is_valid[sub]:
        #       ...
        #   sub = (sub - 1) & remaining
        
        # Optimization: Generate submasks containing v.
        # Let rest = remaining ^ lb.
        # sub = lb | (sub_rest) where sub_rest is a submask of rest.
        rest = remaining ^ lb
        
        # To find larger blocks first, we can iterate submasks in a way that prioritizes larger ones?
        # Standard iteration (sub-1)&mask doesn't guarantee size order.
        # However, for small n, it's fine.
        
        sub_rest = rest
        while True:
            sub = lb | sub_rest
            if is_valid[sub]:
                # Try this block
                new_remaining = remaining ^ sub
                if dfs(new_remaining, current_parts + [sub]):
                    return True # Timeout propagated
            if sub_rest == 0:
                break
            sub_rest = (sub_rest - 1) & rest
            
        return False

    # Start DFS
    full_mask = (1 << n) - 1
    timed_out = dfs(full_mask, [])
    
    # Reconstruct best_parts from masks if it was updated in DFS
    # Wait, in the DFS above, I stored `current_parts` as a list of masks.
    # The `best_parts` update inside DFS copies the list of masks.
    # I need to convert masks to lists of vertices.
    
    if best_parts and isinstance(best_parts[0], int):
        # Convert masks to lists
        final_parts = []
        for m in best_parts:
            verts = []
            for i in range(n):
                if m & (1 << i):
                    verts.append(i)
            final_parts.append(verts)
        best_parts = final_parts

    # Ensure types are correct: list of lists of ints
    # The greedy function returned lists of ints.
    # The DFS update returns lists of ints (after conversion above).
    
    return best_parts
# EVOLVE-BLOCK-END
