import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    edges = P['edges']
    
    # Map edges to indices for bitmask representation
    edge_to_idx = {}
    idx_to_edge = []
    for i, (u, v) in enumerate(edges):
        if u > v:
            u, v = v, u
        edge_to_idx[(u, v)] = i
        idx_to_edge.append((u, v))
        
    m = len(edges)
    
    # Adjacency matrix as bitmasks for vertices
    # adj[u] is a bitmask of vertices v such that (u,v) is an edge
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Function to get edge bitmask for a biclique (L, R)
    def get_edge_mask(L, R):
        mask = 0
        for u in L:
            for v in R:
                if u == v:
                    continue
                key = (u, v) if u < v else (v, u)
                if key in edge_to_idx:
                    mask |= (1 << edge_to_idx[key])
        return mask

    # Generate candidate bicliques
    candidates = []
    seen_bicliques = set()
    
    for u, v in edges:
        if time.time() > deadline:
            break
            
        # Expand (u, v) to maximal biclique
        L = {u}
        R = {v}
        
        changed = True
        while changed:
            changed = False
            
            # Expand L
            new_L = set()
            for x in range(n):
                if x in R:
                    continue
                # Check if x is connected to all in R
                valid = True
                for y in R:
                    if not (adj[x] & (1 << y)):
                        valid = False
                        break
                if valid:
                    new_L.add(x)
            if new_L != L:
                L = new_L
                changed = True
                
            # Expand R
            new_R = set()
            for y in range(n):
                if y in L:
                    continue
                valid = True
                for x in L:
                    if not (adj[x] & (1 << y)):
                        valid = False
                        break
                if valid:
                    new_R.add(y)
            if new_R != R:
                R = new_R
                changed = True
                
        # Normalize for uniqueness
        L_frozen = frozenset(L)
        R_frozen = frozenset(R)
        key = (L_frozen, R_frozen)
        # Also check (R_frozen, L_frozen) just in case, though L and R are distinct by construction (disjoint)
        # Actually L and R are disjoint in our expansion.
        
        if key not in seen_bicliques:
            seen_bicliques.add(key)
            candidates.append((L, R))
            
        # Also add the single edge biclique if not already covered?
        # The maximal biclique contains the edge, so it covers it.
        # But maybe a smaller biclique is better for partition?
        # As argued, maximal is sufficient for cover.
        
    # If no candidates found (shouldn't happen), add single edges
    if not candidates:
        for u, v in edges:
            candidates.append(({u}, {v}))

    # Prepare sets for Set Cover
    # set_masks[i] is the bitmask of edges covered by candidate i
    set_masks = []
    for L, R in candidates:
        set_masks.append(get_edge_mask(L, R))
        
    # Filter out empty sets
    valid_sets = []
    for i, mask in enumerate(set_masks):
        if mask != 0:
            valid_sets.append((i, mask))
            
    # Remove duplicate masks?
    unique_masks = []
    seen_masks = set()
    for i, mask in valid_sets:
        if mask not in seen_masks:
            seen_masks.add(mask)
            unique_masks.append(mask)
            
    # Solve Min Set Cover
    # Universe mask
    universe_mask = (1 << m) - 1
    
    best_cover = []
    best_size = m # Upper bound: each edge is a biclique
    
    # Greedy solution for initial bound
    def greedy_cover():
        uncovered = universe_mask
        cover = []
        while uncovered:
            best_set = None
            best_count = -1
            for mask in unique_masks:
                count = bin(uncovered & mask).count('1')
                if count > best_count:
                    best_count = count
                    best_set = mask
            if best_set is None:
                break
            cover.append(best_set)
            uncovered &= ~best_set
            if time.time() > deadline:
                break
        return cover
        
    init_cover = greedy_cover()
    if len(init_cover) < best_size:
        best_size = len(init_cover)
        best_cover = init_cover[:]
        
    # Branch and Bound
    # To speed up, sort unique_masks by size descending
    unique_masks.sort(key=lambda x: bin(x).count('1'), reverse=True)
    
    # Precompute for each edge index, which sets cover it
    # edge_sets[e] = list of indices in unique_masks that cover edge e
    edge_sets = [[] for _ in range(m)]
    for i, mask in enumerate(unique_masks):
        # Iterate through bits in mask
        m_temp = mask
        while m_temp:
            lsb = m_temp & -m_temp
            idx = lsb.bit_length() - 1
            edge_sets[idx].append(i)
            m_temp ^= lsb
            
    def lower_bound(uncovered):
        # Greedy bound
        lb = 0
        u = uncovered
        while u:
            # Find set with max coverage
            max_cov = 0
            for mask in unique_masks:
                cov = bin(u & mask).count('1')
                if cov > max_cov:
                    max_cov = cov
            if max_cov == 0:
                break
            lb += 1
            # We don't actually remove the best set in a simple LB, 
            # but a tighter LB would. For speed, this is okay.
            # Actually, just counting is not a valid LB.
            # A valid LB is ceil(popcount(uncovered) / max_set_size).
            # Let's use that.
            break
        return (bin(uncovered).count('1') + max_set_size - 1) // max_set_size if max_set_size > 0 else 0
        
    max_set_size = max([bin(m).count('1') for m in unique_masks]) if unique_masks else 1
    
    def bb(uncovered, current_size, current_cover):
        nonlocal best_size, best_cover
        
        if time.time() > deadline:
            return
            
        if current_size >= best_size:
            return
            
        if uncovered == 0:
            if current_size < best_size:
                best_size = current_size
                best_cover = current_cover[:]
            return
            
        # Lower bound
        lb = (bin(uncovered).count('1') + max_set_size - 1) // max_set_size
        if current_size + lb >= best_size:
            return
            
        # Pick an uncovered edge
        # Heuristic: pick edge with fewest sets covering it
        u_temp = uncovered
        best_edge = -1
        min_sets = 1000
        while u_temp:
            lsb = u_temp & -u_temp
            idx = lsb.bit_length() - 1
            if len(edge_sets[idx]) < min_sets:
                min_sets = len(edge_sets[idx])
                best_edge = idx
            u_temp ^= lsb
            
        # Branch on sets covering best_edge
        # Sort by coverage in uncovered
        sets_to_try = edge_sets[best_edge]
        sets_to_try.sort(key=lambda i: bin(uncovered & unique_masks[i]).count('1'), reverse=True)
        
        for i in sets_to_try:
            mask = unique_masks[i]
            new_uncovered = uncovered & ~mask
            current_cover.append(mask)
            bb(new_uncovered, current_size + 1, current_cover)
            current_cover.pop()
            
            if time.time() > deadline:
                return

    bb(universe_mask, 0, [])
    
    # Convert best_cover (list of masks) back to bicliques
    # We need to map masks back to (L, R)
    # Since we lost the mapping when we deduplicated masks, 
    # we should keep a mapping from mask to (L,R)
    
    # Let's redo the candidate generation to keep mapping
    # But for now, if we have the masks, we can reconstruct?
    # No, we need the vertex sets.
    
    # So, we should store (mask, L, R) in unique_masks
    # Let's modify the logic above slightly in the final code.
    
    # For the purpose of this thought process, assume we have the mapping.
    
    # Re-implementing with mapping
    candidates_with_mask = []
    seen_bicliques = set()
    
    for u, v in edges:
        if time.time() > deadline:
            break
        L = {u}
        R = {v}
        changed = True
        while changed:
            changed = False
            new_L = set()
            for x in range(n):
                if x in R: continue
                if all((adj[x] & (1 << y)) for y in R):
                    new_L.add(x)
            if new_L != L:
                L = new_L
                changed = True
            new_R = set()
            for y in range(n):
                if y in L: continue
                if all((adj[x] & (1 << y)) for x in L):
                    new_R.add(y)
            if new_R != R:
                R = new_R
                changed = True
                
        key = (frozenset(L), frozenset(R))
        if key not in seen_bicliques:
            seen_bicliques.add(key)
            mask = get_edge_mask(L, R)
            if mask != 0:
                candidates_with_mask.append((mask, L, R))
                
    # Deduplicate masks
    unique_candidates = []
    seen_masks = set()
    for mask, L, R in candidates_with_mask:
        if mask not in seen_masks:
            seen_masks.add(mask)
            unique_candidates.append((mask, L, R))
            
    # Sort by size
    unique_candidates.sort(key=lambda x: bin(x[0]).count('1'), reverse=True)
    
    # Re-run B&B with this data
    best_cover_masks = []
    best_size = m
    
    # Greedy
    uncovered = universe_mask
    cover = []
    while uncovered:
        best_idx = -1
        best_cov = -1
        for i, (mask, L, R) in enumerate(unique_candidates):
            cov = bin(uncovered & mask).count('1')
            if cov > best_cov:
                best_cov = cov
                best_idx = i
        if best_idx == -1: break
        cover.append(best_idx)
        uncovered &= ~unique_candidates[best_idx][0]
        
    if len(cover) < best_size:
        best_size = len(cover)
        best_cover_masks = cover[:]
        
    # B&B
    edge_sets = [[] for _ in range(m)]
    for i, (mask, L, R) in enumerate(unique_candidates):
        m_temp = mask
        while m_temp:
            lsb = m_temp & -m_temp
            idx = lsb.bit_length() - 1
            edge_sets[idx].append(i)
            m_temp ^= lsb
            
    max_set_size = max([bin(c[0]).count('1') for c in unique_candidates]) if unique_candidates else 1
    
    def bb(uncovered, current_size, current_cover):
        nonlocal best_size, best_cover_masks
        if time.time() > deadline:
            return
        if current_size >= best_size:
            return
        if uncovered == 0:
            if current_size < best_size:
                best_size = current_size
                best_cover_masks = current_cover[:]
            return
        lb = (bin(uncovered).count('1') + max_set_size - 1) // max_set_size
        if current_size + lb >= best_size:
            return
            
        u_temp = uncovered
        best_edge = -1
        min_sets = 1000
        while u_temp:
            lsb = u_temp & -u_temp
            idx = lsb.bit_length() - 1
            if len(edge_sets[idx]) < min_sets:
                min_sets = len(edge_sets[idx])
                best_edge = idx
            u_temp ^= lsb
            
        sets_to_try = edge_sets[best_edge]
        sets_to_try.sort(key=lambda i: bin(uncovered & unique_candidates[i][0]).count('1'), reverse=True)
        
        for i in sets_to_try:
            mask = unique_candidates[i][0]
            new_uncovered = uncovered & ~mask
            current_cover.append(i)
            bb(new_uncovered, current_size + 1, current_cover)
            current_cover.pop()
            if time.time() > deadline:
                return

    bb(universe_mask, 0, [])
    
    # Construct result
    result = []
    for idx in best_cover_masks:
        mask, L, R = unique_candidates[idx]
        result.append([list(L), list(R)])
        
    # If B&B failed or didn't find anything, fallback to greedy
    if not result:
        for idx in best_cover_masks:
            mask, L, R = unique_candidates[idx]
            result.append([list(L), list(R)])
            
    # Fallback if still empty
    if not result:
        for u, v in edges:
            result.append([[u], [v]])
            
    return result

