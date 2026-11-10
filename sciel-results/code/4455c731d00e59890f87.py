"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    non_adj = [0] * n
    for v in range(n):
        non_adj[v] = full_mask & ~adj[v] & ~(1 << v)

    valid_part_masks = set()
    
    for mask in range(1, 1 << n):
        is_clique = True
        is_indep = True
        
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            rest = mask ^ lsb
            if rest & ~adj[v]:
                is_clique = False
            if rest & ~non_adj[v]:
                is_indep = False
            if not is_clique and not is_indep:
                break
            m ^= lsb
            
        if is_clique or is_indep:
            valid_part_masks.add(mask)
            
    parts = [(m, m.bit_count()) for m in valid_part_masks]
    parts.sort(key=lambda x: -x[1])
    
    # Precompute parts containing each vertex, as list of (mask, size)
    parts_by_vertex = [[] for _ in range(n)]
    for mask, size in parts:
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            parts_by_vertex[v].append((mask, size))
            m ^= lsb
            
    best_count = [n + 1]
    best_solution = [None]
    
    def backtrack(remaining_mask, num_parts, current_parts):
        if remaining_mask == 0:
            if num_parts < best_count[0]:
                best_count[0] = num_parts
                best_solution[0] = current_parts.copy()
            return
        
        if num_parts >= best_count[0]:
            return
            
        # Lower bound
        max_size = parts[0][1]
        lb = (remaining_mask.bit_count() + max_size - 1) // max_size
        if num_parts + lb >= best_count[0]:
            return
            
        # Select vertex with fewest compatible parts
        min_count = float('inf')
        chosen_v = -1
        chosen_parts = []
        
        for v in range(n):
            if not (remaining_mask & (1 << v)):
                continue
            
            count = 0
            candidate_parts_for_v = []
            for p_mask, p_size in parts_by_vertex[v]:
                if p_mask & ~remaining_mask:
                    continue
                count += 1
                candidate_parts_for_v.append((p_mask, p_size))
                
            if count < min_count:
                min_count = count
                chosen_v = v
                chosen_parts = candidate_parts_for_v
                
            if min_count == 0:
                break

        if chosen_v == -1:
            return

        # Sort by size descending
        chosen_parts.sort(key=lambda x: -x[1])
        
        for p_mask, _ in chosen_parts:
            new_remaining = remaining_mask & ~p_mask
            current_parts.append(p_mask)
            backtrack(new_remaining, num_parts + 1, current_parts)
            current_parts.pop()
            
    backtrack(full_mask, 0, [])
    
    if best_solution[0] is not None:
        result = []
        for mask in best_solution[0]:
            part = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                part.append(v)
                m ^= lsb
            result.append(part)
        return result
    else:
        return [[v] for v in range(n)]

# EVOLVE-BLOCK-END
