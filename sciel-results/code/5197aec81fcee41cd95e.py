"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list as bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    size = 1 << n
    is_clique = [False] * size
    is_indep = [False] * size
    valid = [False] * size
    
    is_clique[0] = True
    is_indep[0] = True
    valid[0] = True # Empty set is technically both, but we won't use it as a part
    
    # Precompute properties for all masks
    for mask in range(1, size):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: v must be connected to all in rest
        if is_clique[rest]:
            if (adj[v] & rest) == rest:
                is_clique[mask] = True
        
        # Check independent: v must not be connected to any in rest
        if is_indep[rest]:
            if (adj[v] & rest) == 0:
                is_indep[mask] = True
        
        valid[mask] = is_clique[mask] or is_indep[mask]
        
    # DP to find minimum number of parts
    dp = [float('inf')] * size
    dp[0] = 0
    
    deadline = time.time() + 55 # 5s margin for safety
    
    for mask in range(1, size):
        if time.time() > deadline:
            break
            
        if valid[mask]:
            dp[mask] = 1
        else:
            best = float('inf')
            # Iterate submasks of mask
            s = (mask - 1) & mask
            while s:
                if valid[s]:
                    rem = mask ^ s
                    val = 1 + dp[rem]
                    if val < best:
                        best = val
                        if best == 2:
                            break
                s = (s - 1) & mask
            dp[mask] = best
            
    full_mask = size - 1
    min_parts = dp[full_mask]
    
    # Reconstruct the partition
    parts = []
    remaining = full_mask
    while remaining:
        if time.time() > deadline:
            break
        found = False
        s = (remaining - 1) & remaining
        while s:
            if valid[s]:
                rem = remaining ^ s
                if dp[rem] == dp[remaining] - 1:
                    part = []
                    for v in range(n):
                        if s & (1 << v):
                            part.append(v)
                    parts.append(part)
                    remaining = rem
                    found = True
                    break
            s = (s - 1) & remaining
        if not found:
            break
            
    # Verify and fallback
    all_vertices = set()
    for p in parts:
        for v in p:
            all_vertices.add(v)
            
    if all_vertices != set(range(n)) or len(parts) == 0:
        # Fallback: Greedy construction
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                # Check if adding v keeps part as clique or independent
                # Current part is either clique or independent.
                # If clique: v must connect to all in part.
                # If independent: v must not connect to any in part.
                
                # Determine type of part
                is_part_clique = True
                is_part_indep = True
                # We can just check edges between v and part
                # If part is clique, all edges present.
                # If part is independent, no edges present.
                
                # Let's just check if adding v works
                # We need to know if part is currently clique or indep.
                # We can recompute or store. Recomputing is O(k^2) which is small.
                
                # Simpler: Try to add v.
                # If part was clique, check if v connects to all.
                # If part was indep, check if v connects to none.
                
                # Since we don't store type, let's just check consistency.
                # Actually, a set is either clique or indep.
                # If we add v, the new set must be clique or indep.
                
                # Check if part + [v] is clique
                ok_clique = True
                for u in part:
                    if (adj[v] & (1 << u)) == 0:
                        ok_clique = False
                        break
                # Check if part + [v] is indep
                ok_indep = True
                for u in part:
                    if (adj[v] & (1 << u)) != 0:
                        ok_indep = False
                        break
                        
                if ok_clique or ok_indep:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
                
    # Ensure types are correct (list of lists of ints)
    return parts

# EVOLVE-BLOCK-END
