import time
import random

def _construct_polarity_graph(q):
    """
    Construct the polarity graph of the projective plane PG(2, q).
    """
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q-2, q)
                    key = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    key = (0, 1, (z * inv_y) % q)
                else:
                    key = (0, 0, 1)
                
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n_pts = len(points)
    adj = [set() for _ in range(n_pts)]
    
    for i in range(n_pts):
        x1, x2, x3 = points[i]
        for j in range(i + 1, n_pts):
            y1, y2, y3 = points[j]
            if (x1 * y1 + x2 * y2 + x3 * y3) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
                
    edges = []
    for i in range(n_pts):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
                
    return n_pts, edges, adj

def _verify_c4_free(n, adj):
    """Check that no two vertices have >= 2 common neighbours."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return False
    return True

def _has_independent_set(adj_mask, n, limit=11):
    """
    Check if there is an independent set of size >= limit.
    Uses bitmasks for speed.
    adj_mask: list of int bitmasks for neighbors.
    Returns True if an independent set of size >= limit exists.
    """
    all_mask = (1 << n) - 1
    
    # Precompute degree in the full graph? 
    # We need degree in the remaining set.
    
    def _branch(rem_mask, current_size):
        if current_size >= limit:
            return True
        if rem_mask == 0:
            return False
        
        # Upper bound: current_size + popcount(rem_mask)
        if current_size + rem_mask.bit_count() < limit:
            return False
            
        # Heuristic: Pick vertex with max degree in rem_mask
        # Or min degree? 
        # For independent set, picking a vertex with high degree removes more candidates
        # from the "Include" branch, which is good for pruning the "Include" branch.
        # But the "Exclude" branch is the one that keeps the graph large.
        # Standard Bron-Kerbosch for Max Independent Set usually picks max degree.
        
        best_v = -1
        best_deg = -1
        m = rem_mask
        while m:
            v_bit = m & -m
            v = v_bit.bit_length() - 1
            m ^= v_bit
            # Degree of v in rem_mask
            deg = (adj_mask[v] & rem_mask).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
                
        if best_v == -1:
            return False
            
        # Branch 1: Include best_v
        # Remove best_v and its neighbors from rem
        rem_include = rem_mask & ~(adj_mask[best_v] | (1 << best_v))
        if _branch(rem_include, current_size + 1):
            return True
            
        # Branch 2: Exclude best_v
        rem_exclude = rem_mask & ~(1 << best_v)
        if _branch(rem_exclude, current_size):
            return True
            
        return False

    return _branch(all_mask, 0)

def construct():
    deadline = time.time() + 140.0
    
    best_n = 0
    best_edges = []
    
    # Try q=7 (57 vertices) first, as it's the most likely source for n=43 or n=39
    # Then q=5 (31 vertices) as a fallback for n=31
    
    qs_to_try = [7, 5]
    
    for q in qs_to_try:
        if time.time() > deadline - 10:
            break
            
        try:
            n_full, edges_full, adj_full = _construct_polarity_graph(q)
            
            # Verify C4-free
            if not _verify_c4_free(n_full, adj_full):
                continue
                
            # Convert to bitmasks
            adj_mask = [0] * n_full
            for i in range(n_full):
                mask = 0
                for j in adj_full[i]:
                    mask |= (1 << j)
                adj_mask[i] = mask
            
            # Check if the full graph itself has alpha < 11
            # If n_full <= 43, we can potentially use the whole thing
            if n_full <= 43:
                if not _has_independent_set(adj_mask, n_full, limit=11):
                    if n_full > best_n:
                        best_n = n_full
                        best_edges = edges_full
                        if best_n == 43:
                            return {'n': best_n, 'edges': best_edges}
                else:
                    # Full graph invalid, but maybe a subset?
                    # For n=31, if alpha >= 11, we need a smaller subset.
                    # But q=5 usually has small alpha.
                    pass
            
            # If n_full > 43, we need a subset of size 43 (or 39)
            # Let's try to find a subset of size 43 first.
            target_sizes = [43, 42, 41, 40, 39]
            
            for target_n in target_sizes:
                if target_n > n_full:
                    continue
                if target_n <= best_n:
                    continue
                
                if time.time() > deadline - 5:
                    break
                    
                found = False
                attempts = 0
                max_attempts = 5000 # More attempts with fast checker
                
                while attempts < max_attempts and not found:
                    if time.time() > deadline - 2:
                        break
                    attempts += 1
                    
                    # Random subset
                    selected = random.sample(range(n_full), target_n)
                    selected_set = set(selected)
                    
                    # Build subgraph adj_mask
                    # Map old indices to new 0..target_n-1
                    old_to_new = {old: new for new, old in enumerate(selected)}
                    
                    sub_adj_mask = [0] * target_n
                    for old_i in selected:
                        new_i = old_to_new[old_i]
                        # Neighbors of old_i in the subgraph
                        # old_adj_mask[old_i] & selected_mask
                        # But we need to remap bits.
                        # It's faster to iterate neighbors in adj_full[old_i] and check if in selected_set
                        for old_j in adj_full[old_i]:
                            if old_j in selected_set:
                                new_j = old_to_new[old_j]
                                sub_adj_mask[new_i] |= (1 << new_j)
                    
                    # Check alpha
                    if not _has_independent_set(sub_adj_mask, target_n, limit=11):
                        found = True
                        # Build edges
                        candidate_edges = []
                        for i in range(target_n):
                            mask = sub_adj_mask[i]
                            while mask:
                                bit = mask & -mask
                                j = bit.bit_length() - 1
                                if i < j:
                                    candidate_edges.append([i, j])
                                mask ^= bit
                        best_n = target_n
                        best_edges = candidate_edges
                        break
                        
                if found:
                    if best_n == 43:
                        return {'n': best_n, 'edges': best_edges}
                    break # Found a valid size for this q, try next q or stop if 43 found? 
                    # Actually, if we found 39, we might want to try 43 again? 
                    # No, the loop iterates target_sizes descending. So if 43 failed, we try 42, etc.
                    # Once we find a valid one, we record it. 
                    # Should we continue to see if a larger q gives a better result? 
                    # q=7 is first. If we find 39 with q=7, we record it.
                    # We break the target_sizes loop to move to next q? 
                    # If q=5 is next, it only has 31 vertices, so it won't beat 39.
                    # So we can stop.
                    break
                    
        except Exception:
            continue
            
    if best_n == 0:
        # Fallback
        best_n = 3
        best_edges = [[0, 1], [1, 2], [0, 2]]
        
    return {'n': best_n, 'edges': best_edges}
