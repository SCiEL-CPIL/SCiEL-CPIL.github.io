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
    
    size = 1 << n
    
    # Precompute is_clique and is_indep using bit-parallel checks
    is_clique = [False] * size
    is_indep = [False] * size
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, size):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: v must be adjacent to all in rest
        if is_clique[rest]:
            if (rest & ~adj[v]) == 0:
                is_clique[mask] = True
        
        # Check independent: v must not be adjacent to any in rest
        if is_indep[rest]:
            if (rest & adj[v]) == 0:
                is_indep[mask] = True
    
    valid = [is_clique[m] or is_indep[m] for m in range(size)]
    
    # Precompute: for each vertex, list of valid masks containing it
    # This speeds up the branching step
    valid_masks_containing = [[] for _ in range(n)]
    for mask in range(1, size):
        if valid[mask]:
            for v in range(n):
                if mask & (1 << v):
                    valid_masks_containing[v].append(mask)
    
    # Sort masks in each list by size descending to cover more ground early
    for v in range(n):
        valid_masks_containing[v].sort(key=lambda x: bin(x).count('1'), reverse=True)
    
    # Greedy initial solution for upper bound
    def greedy():
        parts = []
        remaining_mask = size - 1
        while remaining_mask:
            # Pick vertex with highest degree in remaining subgraph
            best_v = -1
            best_deg = -1
            for v in range(n):
                if remaining_mask & (1 << v):
                    deg = bin(remaining_mask & adj[v]).count('1')
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
            
            # Try to find largest valid part containing best_v
            best_part_mask = 1 << best_v
            for m in valid_masks_containing[best_v]:
                if m & ~remaining_mask:
                    continue
                if bin(m).count('1') > bin(best_part_mask).count('1'):
                    best_part_mask = m
            
            parts.append(best_part_mask)
            remaining_mask ^= best_part_mask
        return parts
    
    initial_parts = greedy()
    best_count = len(initial_parts)
    best_parts_mask = initial_parts
    
    deadline = time.time() + 55.0  # Safety margin before 60s limit
    
    # Backtracking search for minimum cochromatic partition
    def search(remaining_mask, current_parts, part_count):
        nonlocal best_parts_mask, best_count
        
        if time.time() > deadline:
            return
        
        if not remaining_mask:
            if part_count < best_count:
                best_count = part_count
                best_parts_mask = list(current_parts)
            return
        
        # Pruning: if we can't beat best, return
        if part_count >= best_count:
            return
        
        # Pick vertex with fewest valid placements
        min_v = -1
        min_placements = n + 1
        
        for v in range(n):
            if not (remaining_mask & (1 << v)):
                continue
            # Count valid placements: existing parts + new part
            count = 1  # new part
            for pm in current_parts:
                # Check if adding v to part pm keeps it valid
                trial = pm | (1 << v)
                if valid[trial]:
                    count += 1
            if count < min_placements:
                min_placements = count
                min_v = v
        
        if min_v == -1:
            return
        
        v_bit = 1 << min_v
        new_remaining = remaining_mask ^ v_bit
        
        # Try adding min_v to each existing part
        for i, pm in enumerate(current_parts):
            trial = pm | v_bit
            if valid[trial]:
                new_parts = list(current_parts)
                new_parts[i] = trial
                search(new_remaining, new_parts, part_count)
        
        # Try creating a new part with min_v
        new_parts = list(current_parts)
        new_parts.append(v_bit)
        search(new_remaining, new_parts, part_count + 1)
    
    # Start search
    search(size - 1, [], 0)
    
    # Reconstruct from best_parts_mask
    result = []
    for pm in best_parts_mask:
        part = []
        for v in range(n):
            if pm & (1 << v):
                part.append(v)
        result.append(part)
    
    # Verify and fallback if needed
    all_vertices = set()
    for p in result:
        for v in p:
            all_vertices.add(v)
    if all_vertices != set(range(n)) or len(result) == 0:
        # Fallback: greedy assignment
        result = []
        for v in range(n):
            placed = False
            for part in result:
                trial_mask = 0
                for u in part:
                    trial_mask |= (1 << u)
                trial_mask |= (1 << v)
                if valid[trial_mask]:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                result.append([v])
    
    # Convert to proper types
    final_result = []
    for part in result:
        final_result.append([int(v) for v in part])
    
    return final_result

# EVOLVE-BLOCK-END
