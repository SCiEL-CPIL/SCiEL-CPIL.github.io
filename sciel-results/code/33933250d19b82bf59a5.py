import math
import sys

def construct():
    P_data = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    n = P_data['n']
    edges = P_data['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Order vertices by degree descending to improve pruning?
    # Or just use 0..n-1. Let's try 0..n-1 first for simplicity, 
    # but maybe high degree vertices constrain more.
    # Let's stick to 0..n-1 for now, but we can reorder if needed.
    
    # State:
    # parts_masks: list of int masks
    # part_types: list of int (0: Undecided, 1: Clique, 2: Independent)
    
    best_count = n + 1 # Upper bound
    best_partition = None
    
    # We fix vertex 0 to part 0 to break symmetry
    # But we need to try all possibilities for vertex 0?
    # No, vertex 0 can be in any part, but by symmetry, we can assume it's in part 0.
    # However, part 0 could be Clique or Independent or Undecided.
    # Since a single vertex is both, we start with part 0 as Undecided containing {0}.
    
    # Actually, let's just run DFS from scratch.
    
    def is_clique_valid(mask, v):
        return (mask & adj[v]) == mask
        
    def is_ind_valid(mask, v):
        return (mask & adj[v]) == 0

    def dfs(v_idx, parts_masks, part_types):
        nonlocal best_count, best_partition
        
        if v_idx == n:
            if len(parts_masks) < best_count:
                best_count = len(parts_masks)
                best_partition = (parts_masks[:], part_types[:])
            return
            
        # Pruning: if current parts count + 1 (min new part for v) >= best_count, prune?
        # Not exactly, v can join existing.
        # Lower bound: we need at least 1 part for v if it can't join any.
        # Simple bound: len(parts_masks) >= best_count -> prune.
        if len(parts_masks) >= best_count:
            return
            
        v = v_idx
        v_mask = 1 << v
        
        # Try adding v to existing parts
        for i in range(len(parts_masks)):
            mask = parts_masks[i]
            ptype = part_types[i]
            
            # Check if v can be added
            c_ok = is_clique_valid(mask, v)
            i_ok = is_ind_valid(mask, v)
            
            if ptype == 0: # Undecided
                if c_ok and i_ok:
                    # Can remain Undecided
                    new_parts = parts_masks[:]
                    new_types = part_types[:]
                    new_parts[i] = mask | v_mask
                    new_types[i] = 0
                    dfs(v_idx + 1, new_parts, new_types)
                    
                    # We could also commit to Clique or Ind here?
                    # No, keeping it Undecided is strictly more powerful.
                    # If we commit to Clique, we lose the option to add a non-neighbor later.
                    # So we only keep Undecided if both are possible.
                elif c_ok:
                    # Must be Clique
                    new_parts = parts_masks[:]
                    new_types = part_types[:]
                    new_parts[i] = mask | v_mask
                    new_types[i] = 1
                    dfs(v_idx + 1, new_parts, new_types)
                elif i_ok:
                    # Must be Independent
                    new_parts = parts_masks[:]
                    new_types = part_types[:]
                    new_parts[i] = mask | v_mask
                    new_types[i] = 2
                    dfs(v_idx + 1, new_parts, new_types)
            elif ptype == 1: # Clique
                if c_ok:
                    new_parts = parts_masks[:]
                    new_types = part_types[:]
                    new_parts[i] = mask | v_mask
                    # Type remains Clique
                    dfs(v_idx + 1, new_parts, new_types)
            elif ptype == 2: # Independent
                if i_ok:
                    new_parts = parts_masks[:]
                    new_types = part_types[:]
                    new_parts[i] = mask | v_mask
                    # Type remains Independent
                    dfs(v_idx + 1, new_parts, new_types)
                    
        # Try adding v to a new part
        # Symmetry breaking: if we have empty parts, we only create one.
        # Since we start with no parts, or parts are non-empty.
        # We just append a new part.
        # To break symmetry, if the new part is empty (which it isn't, it has v), 
        # we don't need to worry about swapping empty parts.
        # However, if we have multiple parts that are identical in content and type, 
        # we might explore symmetric branches. 
        # A simple symmetry break: only create a new part if v cannot be added to any existing part 
        # with the same "signature"? That's complex.
        # For n=14, this is likely fine without advanced symmetry breaking.
        
        new_parts = parts_masks[:] + [v_mask]
        new_types = part_types[:] + [0] # Single vertex is Undecided (both)
        dfs(v_idx + 1, new_parts, new_types)

    # Initialize with vertex 0 in part 0
    # We can also just start with empty lists and let the DFS handle it.
    # But fixing vertex 0 reduces search space.
    # Let's fix vertex 0 to part 0, type Undecided.
    dfs(1, [1 << 0], [0])
    
    # If best_partition is None, it means something went wrong, but n=14 should be solvable.
    # Convert best_partition to list of lists
    result = []
    if best_partition:
        masks, _ = best_partition
        for mask in masks:
            part = []
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                part.append(idx)
                m ^= lsb
            result.append(part)
    else:
        # Fallback: single vertex parts
        for i in range(n):
            result.append([i])
            
    return result
