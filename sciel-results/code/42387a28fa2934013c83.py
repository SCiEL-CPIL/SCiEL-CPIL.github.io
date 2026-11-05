import time
import random

# EVOLVE-BLOCK-START
def construct():
    N = 256
    # Vertices are 1..256, we use 0..255 as indices
    # adj[i] is a bitmask of vertices j such that (i+1, j+1) conflict
    adj = [0] * N
    
    # Precompute conflicts
    # (a*b) % (a+b) == 0
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')
        
    def get_set_from_mask(mask):
        return [i + 1 for i in range(N) if (mask >> i) & 1]
        
    best_mask = 0
    best_size = 0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0 # Leave 5s margin
    
    # Local search function: try 2-swaps
    def local_search(mask):
        current_size = popcount(mask)
        improved = True
        while improved:
            improved = False
            # Get list of vertices in mask
            verts = [i for i in range(N) if (mask >> i) & 1]
            # Try all pairs
            for idx_u in range(len(verts)):
                for idx_v in range(idx_u + 1, len(verts)):
                    u = verts[idx_u]
                    v = verts[idx_v]
                    # Remove u and v
                    new_mask = mask & ~(1 << u) & ~(1 << v)
                    # Find candidates that can be added
                    # A vertex w can be added if it is not in new_mask and has no conflict with new_mask
                    candidates = []
                    for w in range(N):
                        if (new_mask >> w) & 1:
                            continue
                        if (adj[w] & new_mask) == 0:
                            candidates.append(w)
                    
                    if len(candidates) <= 2:
                        continue
                    
                    # We need to find the maximum independent set in the subgraph induced by candidates
                    # Since candidates are only those that don't conflict with new_mask,
                    # we just need MIS of the graph formed by conflicts among candidates.
                    # This subgraph is likely small.
                    # Use a simple greedy or brute force if very small
                    # For now, greedy: sort by degree in subgraph, add if possible
                    cand_set = set(candidates)
                    # Build sub-adjacency for candidates
                    sub_adj = {}
                    for w in candidates:
                        neighbors = [x for x in candidates if x != w and (adj[w] >> x) & 1]
                        sub_adj[w] = neighbors
                    
                    # Greedy MIS on subgraph
                    # Sort by degree ascending
                    sorted_cands = sorted(candidates, key=lambda x: len(sub_adj[x]))
                    local_mask = 0
                    for w in sorted_cands:
                        if (local_mask & adj[w]) == 0: # Check against already picked in local
                            local_mask |= (1 << w)
                            
                    local_size = popcount(local_mask)
                    if local_size > 2:
                        new_total_mask = new_mask | local_mask
                        new_total_size = popcount(new_total_mask)
                        if new_total_size > current_size:
                            mask = new_total_mask
                            current_size = new_total_size
                            improved = True
                            # Restart pair loop with new mask
                            break
                    if improved:
                        break
        return mask, current_size

    # Main Loop: Randomized Greedy + Local Search
    while time.time() < deadline:
        # Randomized Greedy Construction
        order = list(range(N))
        random.shuffle(order)
        
        # Option 1: Simple Greedy
        current_mask = 0
        for v in order:
            if (adj[v] & current_mask) == 0:
                current_mask |= (1 << v)
                
        current_size = popcount(current_mask)
        
        # Local Search
        current_mask, current_size = local_search(current_mask)
        
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask
            
        # If we found a very large set, we might stop early or just continue to polish
        # Theoretical max is 256, but likely around 180-200
        
    return get_set_from_mask(best_mask)
# EVOLVE-BLOCK-END
