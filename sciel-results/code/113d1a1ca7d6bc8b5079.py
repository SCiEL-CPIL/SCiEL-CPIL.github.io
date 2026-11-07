"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute is_clique and is_ind for all subsets
    # is_clique[mask]: True if mask induces a clique
    # is_ind[mask]: True if mask induces an independent set
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    is_clique[0] = True
    is_ind[0] = True
    
    for mask in range(1, 1 << n):
        # Get least significant bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: v must be connected to all in rest, and rest must be clique
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
            
        # Check independent set: v must not be connected to any in rest, and rest must be independent
        if is_ind[rest] and (adj[v] & rest) == 0:
            is_ind[mask] = True
            
    is_valid = [is_clique[m] or is_ind[m] for m in range(1 << n)]
    
    # Helper to convert mask to list of vertices
    def mask_to_verts(mask):
        return [i for i in range(n) if (mask >> i) & 1]
    
    # Fallback greedy solution
    def greedy_solution():
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial_mask = 0
                for u in part:
                    trial_mask |= (1 << u)
                trial_mask |= (1 << v)
                if is_valid[trial_mask]:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts

    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    best_solution = greedy_solution()
    
    # DP to find minimum number of parts
    # min_parts[mask] = minimum number of valid parts needed to cover mask
    INF = n + 1
    min_parts = [INF] * (1 << n)
    min_parts[0] = 0
    
    # We iterate masks in increasing order.
    # For each mask, we try to remove a valid submask 'sub' containing the least significant set bit of mask.
    # This ensures we don't double count permutations of parts.
    # Specifically, if v is the LSB of mask, we only consider submasks 'sub' of mask that contain v.
    # This is a standard technique for set partitioning DP to break symmetry.
    
    # Precompute list of valid submasks for each mask? No, too much memory.
    # Just iterate submasks on the fly.
    
    # To speed up, we can precompute for each mask, the list of valid submasks containing its LSB.
    # But 2^14 is small enough to just do it in the loop.
    
    # Let's iterate mask from 1 to full_mask.
    # For each mask, let v = lsb(mask).
    # Iterate sub = mask, (sub-1)&mask, ...
    # If (sub & v_bit) and is_valid[sub]:
    #    min_parts[mask] = min(min_parts[mask], 1 + min_parts[mask ^ sub])
    
    # This is 3^n operations roughly. 3^14 ~ 4.8M. Very fast.
    
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            break
            
        v_bit = mask & -mask
        # Iterate submasks of mask
        sub = mask
        best_val = INF
        
        while sub:
            if (sub & v_bit) and is_valid[sub]:
                rem = mask ^ sub
                val = 1 + min_parts[rem]
                if val < best_val:
                    best_val = val
            sub = (sub - 1) & mask
            
        min_parts[mask] = best_val
        
    # If we completed the DP
    if min_parts[full_mask] < INF:
        k = min_parts[full_mask]
        
        # Reconstruct the partition
        # We need to find the specific submasks used.
        # We can re-run the logic to pick one optimal submask for each step.
        
        parts = []
        rem = full_mask
        while rem:
            v_bit = rem & -rem
            # Find a valid submask 'sub' of rem containing v_bit such that 1 + min_parts[rem ^ sub] == min_parts[rem]
            sub = rem
            found = False
            while sub:
                if (sub & v_bit) and is_valid[sub]:
                    if 1 + min_parts[rem ^ sub] == min_parts[rem]:
                        parts.append(mask_to_verts(sub))
                        rem ^= sub
                        found = True
                        break
                sub = (sub - 1) & rem
            
            if not found:
                # Should not happen if DP is correct
                break
        
        if rem == 0 and len(parts) == k:
            # Verify solution just in case
            all_verts = []
            for part in parts:
                all_verts.extend(part)
            if sorted(all_verts) == list(range(n)):
                valid = True
                for part in parts:
                    mask = 0
                    for v in part:
                        mask |= (1 << v)
                    if mask not in [m for m in range(1<<n) if is_valid[m]]: # This check is redundant but safe
                        valid = False
                        break
                    # Actually just check is_valid[mask]
                    if not is_valid[mask]:
                        valid = False
                        break
                if valid:
                    return parts
    
    return best_solution
# EVOLVE-BLOCK-END
