import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_sets():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Find all maximal cliques
    maximal_cliques = []
    
    def is_clique(mask):
        for u in range(n):
            if mask & (1 << u):
                others = mask & ~(1 << u)
                if others & ~adj[u]:
                    return False
        return True
    
    def is_independent(mask):
        for u in range(n):
            if mask & (1 << u):
                others = mask & ~(1 << u)
                if others & adj[u]:
                    return False
        return True
    
    # Enumerate all subsets and find maximal cliques and independent sets
    all_cliques = []
    all_independent = []
    
    for mask in range(1, 1 << n):
        if is_clique(mask):
            all_cliques.append(mask)
        if is_independent(mask):
            all_independent.append(mask)
    
    # Find maximal cliques: a clique is maximal if no superset is also a clique
    for mask in all_cliques:
        is_maximal = True
        for other in all_cliques:
            if other != mask and (mask | other) == other:
                is_maximal = False
                break
        if is_maximal:
            maximal_cliques.append(mask)
    
    # Find maximal independent sets
    maximal_independent = []
    for mask in all_independent:
        is_maximal = True
        for other in all_independent:
            if other != mask and (mask | other) == other:
                is_maximal = False
                break
        if is_maximal:
            maximal_independent.append(mask)
    
    # Combine: all valid parts are maximal cliques or maximal independent sets
    # But we also need non-maximal ones potentially? No, for optimal partition we can use maximal
    # Actually, we should include all valid masks for flexibility, but start with maximal
    # Let's include all valid masks sorted by size descending
    valid_parts = all_cliques + all_independent
    # Remove duplicates
    valid_parts = list(set(valid_parts))
    # Sort by size descending (larger parts first for better pruning)
    valid_parts.sort(key=lambda m: bin(m).count('1'), reverse=True)
    
    return adj, full_mask, valid_parts

def _greedy_partition(adj, full_mask, valid_parts):
    """Greedy: repeatedly pick the largest valid part that fits."""
    remaining = full_mask
    parts = []
    while remaining:
        best_part = None
        best_size = -1
        for p in valid_parts:
            if p & remaining == p:
                size = bin(p).count('1')
                if size > best_size:
                    best_size = size
                    best_part = p
        if best_part is None:
            # Should not happen; take single vertex
            v = (remaining & -remaining).bit_length() - 1
            best_part = 1 << v
        parts.append(best_part)
        remaining ^= best_part
    return parts

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin
    
    adj, full_mask, valid_parts = _build_sets()
    
    # Get initial upper bound from greedy
    initial_parts = _greedy_partition(adj, full_mask, valid_parts)
    best_count = len(initial_parts)
    best_parts = initial_parts[:]
    
    # Branch and bound
    # State: remaining vertices, current parts list, count
    # Try to find a partition with fewer parts
    
    def can_cover(remaining, parts_count, max_parts):
        """Check if remaining can be covered in at most max_parts parts."""
        if remaining == 0:
            return True
        if parts_count >= max_parts:
            return False
        # Lower bound: minimum number of parts needed
        # Each part can cover at most max_part_size vertices
        max_part_size = 0
        for p in valid_parts:
            if p & remaining == p:
                size = bin(p).count('1')
                if size > max_part_size:
                    max_part_size = size
        if max_part_size == 0:
            return False
        import math
        lb = math.ceil(bin(remaining).count('1') / max_part_size)
        if parts_count + lb > max_parts:
            return False
        return True
    
    def search(remaining, parts_list, depth):
        nonlocal best_count, best_parts, deadline
        
        if time.time() > deadline:
            return
        
        if remaining == 0:
            if len(parts_list) < best_count:
                best_count = len(parts_list)
                best_parts = parts_list[:]
            return
        
        # Prune: if current depth already >= best_count, no improvement possible
        if depth >= best_count:
            return
        
        # Try each valid part that fits in remaining
        # Order by size descending for better pruning
        for p in valid_parts:
            if p & remaining == p:
                # Check if this choice can still lead to improvement
                new_remaining = remaining ^ p
                # Lower bound on additional parts needed
                # Simple: at least 1 if new_remaining != 0
                if new_remaining != 0 and depth + 1 >= best_count:
                    continue
                
                parts_list.append(p)
                search(new_remaining, parts_list, depth + 1)
                parts_list.pop()
                
                if time.time() > deadline:
                    return
    
    # Start search
    search(full_mask, [], 0)
    
    # Convert best_parts to result format
    result = []
    for p in best_parts:
        part_verts = []
        for i in range(P['n']):
            if p & (1 << i):
                part_verts.append(int(i))
        result.append(part_verts)
    
    return result
