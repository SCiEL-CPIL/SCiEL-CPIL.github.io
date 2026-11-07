"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask for each vertex
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute validity of all subsets using bitmask
    # A subset is valid if it's a clique or an independent set
    total_masks = 1 << n
    valid = [False] * total_masks
    valid[0] = True
    
    # For each mask, check if it's a clique or independent set
    for mask in range(1, total_masks):
        # Get vertices in mask
        verts = []
        for i in range(n):
            if mask & (1 << i):
                verts.append(i)
        
        if len(verts) <= 1:
            valid[mask] = True
            continue
        
        # Check if clique: every pair must be adjacent
        is_clique = True
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                u, v = verts[i], verts[j]
                if (adj[u] & (1 << v)) == 0:
                    is_clique = False
                    break
            if not is_clique:
                break
        
        if is_clique:
            valid[mask] = True
            continue
        
        # Check if independent: no pair should be adjacent
        is_indep = True
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                u, v = verts[i], verts[j]
                if (adj[u] & (1 << v)) != 0:
                    is_indep = False
                    break
            if not is_indep:
                break
        
        if is_indep:
            valid[mask] = True
    
    # Collect all valid masks (non-empty)
    all_valid_masks = [mask for mask in range(1, total_masks) if valid[mask]]
    
    # For each vertex, precompute list of valid masks containing it, sorted by size descending
    valid_masks_for_vertex = [[] for _ in range(n)]
    for mask in all_valid_masks:
        for v in range(n):
            if mask & (1 << v):
                valid_masks_for_vertex[v].append(mask)
    
    # Sort by size descending to try larger parts first (better for minimizing count)
    for v in range(n):
        valid_masks_for_vertex[v].sort(key=lambda m: bin(m).count('1'), reverse=True)
    
    # Greedy initialization to get an upper bound
    def greedy_init():
        parts = []
        remaining = set(range(n))
        while remaining:
            # Pick vertex with highest degree in remaining
            v = max(remaining, key=lambda x: sum(1 for y in remaining if y != x and (adj[x] & (1 << y))))
            best_part = [v]
            for u in sorted(remaining - {v}):
                trial = best_part + [u]
                # Check if trial is clique or independent
                is_c = True
                is_i = True
                for i in range(len(trial)):
                    for j in range(i + 1, len(trial)):
                        a, b = trial[i], trial[j]
                        if (adj[a] & (1 << b)) == 0:
                            is_c = False
                        else:
                            is_i = False
                        if not is_c and not is_i:
                            break
                    if not is_c and not is_i:
                        break
                if is_c or is_i:
                    best_part.append(u)
            parts.append(best_part)
            remaining -= set(best_part)
        return parts
    
    initial_parts = greedy_init()
    best_count = len(initial_parts)
    best_partition = initial_parts
    
    # Branch and bound search
    target = (1 << n) - 1
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    def solve(rem_mask, current_count, current_partition):
        nonlocal best_count, best_partition
        
        if time.time() > deadline:
            return
        
        if rem_mask == 0:
            if current_count < best_count:
                best_count = current_count
                best_partition = [list(p) for p in current_partition]
            return
        
        # Pruning: if we can't beat best, return
        if current_count >= best_count:
            return
        
        # Lower bound: at least 1 more part needed
        if current_count + 1 >= best_count and rem_mask != 0:
            # We need at least one more part, so if current_count + 1 >= best_count, prune
            # But we might be able to do it in exactly best_count - current_count parts
            # So only prune if current_count + 1 > best_count, i.e., current_count >= best_count
            # Already handled above. Let's add a stronger bound:
            # The minimum number of parts needed is at least the size of the largest clique
            # or the size of the largest independent set in the remaining subgraph, but that's complex.
            # Simple bound: each part can cover at most max_part_size vertices
            # For now, just use the simple bound above.
            pass
        
        # Pick the vertex with the fewest valid mask options in rem_mask
        # This is a good heuristic for constraint propagation
        min_v = -1
        min_options = float('inf')
        for v in range(n):
            if rem_mask & (1 << v):
                count = 0
                for mask in valid_masks_for_vertex[v]:
                    if (mask & rem_mask) == mask:
                        count += 1
                        if count >= min_options:
                            break
                if count < min_options:
                    min_options = count
                    min_v = v
                    if count == 0:
                        break
        
        if min_v == -1:
            # No vertices remaining? Shouldn't happen if rem_mask != 0
            return
        
        # Try all valid masks containing min_v that are subsets of rem_mask
        for mask in valid_masks_for_vertex[min_v]:
            if (mask & rem_mask) != mask:
                continue
            
            # Check if this mask is valid (should be by construction, but double-check)
            if not valid[mask]:
                continue
            
            new_rem = rem_mask ^ mask
            current_partition.append([v for v in range(n) if mask & (1 << v)])
            solve(new_rem, current_count + 1, current_partition)
            current_partition.pop()
            
            # Early termination if we found optimal (1 part)
            if best_count == 1:
                return
    
    # Run B&B search
    solve(target, 0, [])
    
    # Convert best_partition to proper format
    if best_partition is None:
        best_partition = initial_parts
    
    result = []
    for part in best_partition:
        result.append(list(part))
    
    return result

# EVOLVE-BLOCK-END
