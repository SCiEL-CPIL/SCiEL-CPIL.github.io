import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 90.0
    
    N = 24
    neighbors = [0] * (N + 1)
    
    # Precompute reciprocals
    recips = {i: Fraction(1, i) for i in range(1, N + 1)}
    
    # For each x, for each y > x, check if y conflicts with x
    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            if time.time() > deadline - 20:
                break
                
            target = recips[x] - recips[y]
            if target <= 0:
                continue
                
            # Available elements: {x+1, ..., 24} \ {y}
            available = [z for z in range(x + 1, N + 1) if z != y]
            
            # Check if target can be formed by subset of available
            # Use DFS
            found = False
            
            # Sort available by value? 
            # Smaller denominators give larger fractions.
            # To find a sum quickly, maybe try larger fractions first?
            
            def dfs(idx, remaining):
                nonlocal found
                if found:
                    return
                if remaining == 0:
                    found = True
                    return
                if idx == len(available) or remaining < 0:
                    return
                    
                # Pruning: if sum of all remaining possible < remaining
                # We can precompute suffix sums of available reciprocals
                
                # Include available[idx]
                val = recips[available[idx]]
                if val <= remaining:
                    dfs(idx + 1, remaining - val)
                    
                # Exclude available[idx]
                dfs(idx + 1, remaining)
                
            # Optimize: Precompute suffix sums for pruning
            suffix_sums = [Fraction(0)] * (len(available) + 1)
            for i in range(len(available) - 1, -1, -1):
                suffix_sums[i] = suffix_sums[i + 1] + recips[available[i]]
                
            def dfs_opt(idx, remaining):
                nonlocal found
                if found:
                    return
                if remaining == 0:
                    found = True
                    return
                if idx == len(available):
                    return
                if remaining < 0:
                    return
                if suffix_sums[idx] < remaining:
                    return
                    
                val = recips[available[idx]]
                if val <= remaining:
                    dfs_opt(idx + 1, remaining - val)
                dfs_opt(idx + 1, remaining)
                
            dfs_opt(0, target)
            
            if found:
                neighbors[x] |= (1 << (y - 1))
                # Since graph is undirected for MIS, we should also mark y?
                # No, the conflict is symmetric: "x and y cannot be together".
                # So we just need the edge.
                # neighbors[x] has y.
                # We don't strictly need neighbors[y] to have x for the B&B if we handle it carefully,
                # but standard MIS B&B uses an adjacency matrix or list.
                # Let's make it symmetric to be safe for any B&B implementation.
                neighbors[y] |= (1 << (x - 1))
                
    # Solve MIS
    best_set = 0
    best_size = 0
    
    def branch_and_bound(candidates, current_size, current_mask):
        nonlocal best_size, best_set
        if time.time() > deadline - 5:
            return
            
        if current_size + candidates.bit_count() <= best_size:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_mask
            return
            
        # Pick a vertex
        # Heuristic: Pick vertex with highest degree in candidates?
        # For speed, just pick first.
        v_bit = candidates & -candidates
        v = v_bit.bit_length() - 1
        v_idx = v + 1
        
        # Include v
        new_candidates = candidates & ~v_bit & ~neighbors[v_idx]
        branch_and_bound(new_candidates, current_size + 1, current_mask | v_bit)
        
        # Exclude v
        new_candidates = candidates & ~v_bit
        branch_and_bound(new_candidates, current_size, current_mask)
        
    all_vertices_mask = (1 << N) - 1
    branch_and_bound(all_vertices_mask, 0, 0)
    
    result = []
    for i in range(N):
        if (best_set >> i) & 1:
            result.append(i + 1)
            
    if not result:
        result = list(range(1, 25))
        
    return result
