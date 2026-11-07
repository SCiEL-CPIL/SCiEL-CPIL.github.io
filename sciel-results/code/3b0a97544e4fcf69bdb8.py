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
    
    # Memoization for the search: min_parts[rem] = minimum number of parts needed to cover 'rem'
    # We initialize with a large value
    INF = n + 1
    min_parts = [INF] * (1 << n)
    min_parts[0] = 0
    
    # Also store the actual partition for reconstruction
    # solution[rem] = the best first_part_mask for rem
    solution = [None] * (1 << n)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Iterative DP: iterate rem from 1 to 2^n - 1
    # For each rem, try all non-empty submasks sub of rem that are valid
    # min_parts[rem] = 1 + min(min_parts[rem ^ sub]) over valid sub
    
    for rem in range(1, 1 << n):
        if time.time() > deadline:
            break
            
        # Quick check: if rem itself is valid, answer is 1
        if is_valid[rem]:
            min_parts[rem] = 1
            solution[rem] = rem
            continue
            
        # Find the best submask
        sub = rem
        best_val = INF
        best_sub = None
        
        while sub:
            if is_valid[sub]:
                rest = rem ^ sub
                val = min_parts[rest] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
                    if best_val == 2:  # Can't do better than 2 since rem is not valid
                        break
            sub = (sub - 1) & rem
            
        min_parts[rem] = best_val
        solution[rem] = best_sub
    
    # Reconstruct the solution from full_mask
    if min_parts[full_mask] < INF:
        parts = []
        rem = full_mask
        while rem:
            sub = solution[rem]
            if sub is None:
                break
            # Convert mask to list of vertices
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
        if len(parts) == min_parts[full_mask]:
            all_verts = []
            for part in parts:
                all_verts.extend(part)
            if sorted(all_verts) == list(range(n)):
                # Ensure types are correct: list of lists of int
                return [[int(v) for v in part] for part in parts]
    
    # Fallback greedy solution
    def greedy_solution():
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial = part + [v]
                # Check if trial is clique or independent
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
