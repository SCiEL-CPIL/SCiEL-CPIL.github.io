import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    n = 16
    # 1. Precompute Q3 masks
    q3_edges_local = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    
    bit_index_map = {}
    idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            bit_index_map[(i, j)] = idx
            idx += 1
            
    q3_masks_set = set()
    for perm in itertools.permutations(range(8)):
        mask = 0
        for u, v in q3_edges_local:
            pu = perm[u]
            pv = perm[v]
            if pu > pv:
                pu, pv = pv, pu
            mask |= (1 << bit_index_map[(pu, pv)])
        q3_masks_set.add(mask)
        
    q3_masks_list = list(q3_masks_set)
    q3_masks_np = np.array(q3_masks_list, dtype=np.uint32)
    
    # 2. Precompute Subsets and Edge Positions
    subsets = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets)
    
    # For each edge (u,v), store list of (subset_idx, bit_idx)
    edge_to_subsets = [[[] for _ in range(n)] for _ in range(n)]
    
    # For each subset, store edge mapping (u,v) -> bit_idx
    subset_edges = []
    
    for s_idx, combo in enumerate(subsets):
        edge_dict = {}
        for i in range(8):
            for j in range(i+1, 8):
                u = combo[i]
                v = combo[j]
                b_idx = bit_index_map[(i, j)]
                edge_dict[(u, v)] = b_idx
                edge_to_subsets[u][v].append((s_idx, b_idx))
                edge_to_subsets[v][u].append((s_idx, b_idx))
        subset_edges.append(edge_dict)
        
    # 3. Greedy Initialization (from current program, simplified)
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
    # Initial masks
    current_masks = np.zeros(num_subsets, dtype=np.uint32)
    for s_idx in range(num_subsets):
        mask = 0
        edge_dict = subset_edges[s_idx]
        for (u, v), b_idx in edge_dict.items():
            if (adj[u] >> v) & 1:
                mask |= (1 << b_idx)
        current_masks[s_idx] = mask
        
    # Check initial validity (should be invalid, so we remove edges)
    # We use the current program's removal logic to get a valid start
    # But we need to update current_masks
    
    def has_q3_fast(mask_uint32, q3_masks_np):
        # Check if mask_uint32 contains any q3 mask
        # mask_uint32 is a scalar
        # q3_masks_np is an array
        # (mask & q3) == q3
        # Return True if any
        return np.any((mask_uint32 & q3_masks_np) == q3_masks_np)
    
    # Remove edges to make it valid
    # We'll use a simple greedy removal
    # Find a violating subset
    # To find violations, we can check all subsets
    # This is slow, but we only do it once or twice
    
    # Optimization: Just use the current program's removal logic but update current_masks
    # Actually, let's just run the local search from a random valid graph?
    # Or just start with K16 and remove edges using the fast check?
    
    # Let's try to remove edges from K16 using the fast check
    # We need to find a Q3.
    # Iterate subsets, check if has_q3.
    
    # To speed up initial cleanup:
    # Pick a random subset, if it has Q3, remove a random edge from it.
    # Repeat until no Q3.
    
    max_cleanup_steps = 1000
    for _ in range(max_cleanup_steps):
        if time.time() - start_time > time_limit - 30:
            break
            
        # Find a violating subset
        viol_idx = -1
        # Randomly check subsets to find a violation faster?
        # Or just iterate? 12870 checks * 840 ops = 10M ops. ~0.1s?
        # Let's just iterate.
        for s_idx in range(num_subsets):
            if has_q3_fast(current_masks[s_idx], q3_masks_np):
                viol_idx = s_idx
                break
                
        if viol_idx == -1:
            break
            
        combo = subsets[viol_idx]
        edge_dict = subset_edges[viol_idx]
        
        # Remove a random edge from this subset that is present in the graph
        edges_in_subset = []
        for (u, v), b_idx in edge_dict.items():
            if (adj[u] >> v) & 1:
                edges_in_subset.append((u, v, b_idx))
                
        if not edges_in_subset:
            continue
            
        u, v, b_idx = random.choice(edges_in_subset)
        
        # Remove edge (u, v)
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        
        # Update masks
        for (s_idx2, b_idx2) in edge_to_subsets[u][v]:
            current_masks[s_idx2] ^= (1 << b_idx2)
            
    # Now we have a valid graph.
    best_edge_count = sum(adj[i].bit_count() for i in range(n)) // 2
    best_adj = adj[:]
    
    # 4. Local Search
    # Try to add edges
    while time.time() - start_time < time_limit - 3:
        # Pick a random non-edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        if (adj[u] >> v) & 1:
            continue
            
        # Check if adding (u, v) creates a Q3
        # Get affected subset indices
        affected = edge_to_subsets[u][v]
        
        # Extract current masks for these subsets
        # We need to simulate the addition
        # Get indices
        indices = [s_idx for s_idx, b_idx in affected]
        bits = [1 << b_idx for s_idx, b_idx in affected]
        
        # Get current masks
        curr_masks = current_masks[indices]
        
        # New masks
        new_masks = curr_masks | np.array(bits, dtype=np.uint32)
        
        # Check validity
        # (new_masks[:, None] & q3_masks_np[None, :]) == q3_masks_np[None, :]
        # This creates a (len(indices), 840) array
        is_valid = True
        if len(new_masks) > 0:
            # Check if any of the new masks contains a Q3
            # We only need to check if the *new* masks are bad.
            # If a mask was already bad, it would have been caught earlier (we assume current state is valid).
            # So we just check if any new_mask is bad.
            # (new_masks[:, None] & q3_masks_np[None, :]) == q3_masks_np[None, :]
            # np.any(...)
            # To avoid large memory allocation if possible, but 3000*840 is fine.
            check_arr = (new_masks[:, None] & q3_masks_np[None, :]) == q3_masks_np[None, :]
            if np.any(check_arr):
                is_valid = False
                
        if is_valid:
            # Accept
            for (s_idx2, b_idx2) in affected:
                current_masks[s_idx2] |= (1 << b_idx2)
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            new_edge_count = sum(adj[i].bit_count() for i in range(n)) // 2
            if new_edge_count > best_edge_count:
                best_edge_count = new_edge_count
                best_adj = adj[:]
        else:
            # Try a swap?
            # For now, just reject.
            pass

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
