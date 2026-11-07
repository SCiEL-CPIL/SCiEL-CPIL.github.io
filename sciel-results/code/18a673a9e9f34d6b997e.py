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
    size = 1 << n
    
    # Precompute is_clique and is_ind for all subsets
    is_clique = [False] * size
    is_ind = [False] * size
    is_clique[0] = True
    is_ind[0] = True
    
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
            
        if is_ind[rest] and (adj[v] & rest) == 0:
            is_ind[mask] = True
            
    is_valid = [is_clique[m] or is_ind[m] for m in range(size)]
    
    # DP
    INF = n + 1
    dp = [INF] * size
    dp[0] = 0
    prev = [None] * size
    
    start_time = time.time()
    deadline = start_time + 50.0
    
    for mask in range(1, size):
        if time.time() > deadline:
            break
            
        best_val = INF
        best_sub = None
        
        # Iterate submasks of mask
        sub = mask
        while sub:
            if is_valid[sub]:
                rest = mask ^ sub
                val = dp[rest] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
                    if best_val == 1:
                        break
            sub = (sub - 1) & mask
            
        dp[mask] = best_val
        prev[mask] = best_sub
        
    # Reconstruct
    if dp[full_mask] < INF:
        parts = []
        rem = full_mask
        while rem:
            sub = prev[rem]
            if sub is None:
                # Should not happen if dp is valid
                break
            verts = []
            temp = sub
            while temp:
                b = temp & -temp
                idx = b.bit_length() - 1
                verts.append(idx)
                temp ^= b
            parts.append(verts)
            rem ^= sub
            
        # Verify
        if len(parts) == dp[full_mask]:
            all_verts = []
            for part in parts:
                all_verts.extend(part)
            if sorted(all_verts) == list(range(n)):
                return parts
    
    # Fallback greedy (should not be reached for this small graph)
    def greedy_solution():
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial = part + [v]
                is_c = True
                is_i = True
                for i in range(len(trial)):
                    for j in range(i+1, len(trial)):
                        if adj[trial[i]] & (1 << trial[j]):
                            is_i = False
                        else:
                            is_c = False
                        if not is_c and not is_i:
                            break
                    if not is_c and not is_i:
                        break
                if is_c or is_i:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts

    return greedy_solution()
# EVOLVE-BLOCK-END
