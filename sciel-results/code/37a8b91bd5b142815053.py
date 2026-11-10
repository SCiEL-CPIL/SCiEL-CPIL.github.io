import time
import random
import itertools

def _is_k4_free_mask(adj_mask, n):
    """Check if graph is K4-free using bitmasks."""
    for a in range(n):
        for b in range(a+1, n):
            if not (adj_mask[a] >> b) & 1:
                continue
            # Common neighbors of a and b
            common = adj_mask[a] & adj_mask[b]
            # For each c in common, check if c has any neighbor in common
            # If yes, a,b,c,neighbor form a K4
            c_bits = common
            while c_bits:
                lb = c_bits & (-c_bits)
                c = lb.bit_length() - 1
                if adj_mask[c] & common:
                    return False
                c_bits ^= lb
    return True

def _is_k4_free_after_add(adj_mask, n, u, v):
    """Fast check: after adding edge (u,v), is graph still K4-free?
    Only need to check K4s containing the new edge (u,v).
    A K4 containing edge (u,v) has two other vertices c,d that are:
    - common neighbors of u and v (i.e., adjacent to both u and v)
    - c and d are adjacent to each other
    So: among common neighbors of u,v, is there an edge?"""
    common = adj_mask[u] & adj_mask[v]
    c_bits = common
    while c_bits:
        lb = c_bits & (-c_bits)
        c = lb.bit_length() - 1
        if adj_mask[c] & common:
            return False
        c_bits ^= lb
    return True

def _max_tf_size_exact(adj_mask, n, timeout=0.5):
    """
    Find the largest triangle-free induced subgraph size using branch and bound.
    Optimized for n=18.
    """
    start_time = time.time()
    
    # Precompute for each vertex v, the mask of vertices that form a triangle with v and some other vertex?
    # No, simpler: 
    # A set S is triangle-free if for all u,v in S, N(u) \cap N(v) \cap S is empty.
    
    # We will try to find if there is a TF set of size k, starting from n down to 1.
    # To speed up, we can use a greedy upper bound or just check combinations with pruning.
    
    # For n=18, checking C(18, 15) = 816 combinations is fast.
    # Checking C(18, 16) = 153.
    # Checking C(18, 17) = 18.
    
    def is_triangle_free(subset_mask):
        """Check if the set of vertices in subset_mask induces a triangle-free graph."""
        # Iterate through vertices in subset
        v_bits = subset_mask
        while v_bits:
            lb = v_bits & (-v_bits)
            u = lb.bit_length() - 1
            v_bits ^= lb
            # Neighbors of u within the subset
            neigh_u_in_sub = adj_mask[u] & subset_mask
            # For each neighbor w of u in sub, check if any neighbor of w in sub is also neighbor of u?
            # Actually, we just need to check if N(u) \cap N(w) is empty for all w in N(u) \cap S.
            # More efficiently:
            # Remove u from consideration for other pairs? No, just check all pairs.
            # Let's stick to checking all pairs in the subset.
            pass
        
        # Efficient check:
        # For each edge (u,v) in the induced subgraph, check if they have a common neighbor in the subset.
        # If yes, triangle exists.
        
        # Get list of vertices in subset for easier iteration if needed, but bitmask is better.
        # Let's iterate over edges in the induced subgraph.
        # An edge (u,v) is in the induced subgraph if u,v in subset and adj[u][v] is True.
        
        # Optimization:
        # For each u in subset:
        #   Let N_u = adj_mask[u] & subset_mask
        #   If there is any edge within N_u, then triangle exists.
        #   Check if N_u is an independent set.
        
        v_bits = subset_mask
        while v_bits:
            lb = v_bits & (-v_bits)
            u = lb.bit_length() - 1
            v_bits ^= lb
            
            N_u = adj_mask[u] & subset_mask
            # Check if N_u is independent
            # Iterate through vertices in N_u
            w_bits = N_u
            while w_bits:
                lb2 = w_bits & (-w_bits)
                w = lb2.bit_length() - 1
                w_bits ^= lb2
                # Check if w has any neighbor in N_u
                if adj_mask[w] & N_u:
                    return False
        return True

    # Try sizes from n down to 1
    for k in range(n, 0, -1):
        if time.time() - start_time > timeout:
            # If timeout, return a conservative estimate or the last known good?
            # For the search, we want to know if it's < 15.
            # If we can't decide, we assume it's large (bad) to be safe?
            # Or we can just return the max found so far?
            # Let's return a value that indicates "unknown" or just break.
            # In the context of the search, we want to minimize.
            # If we can't compute exactly, we might miss improvements.
            # But for k >= 15, the number of subsets is small, so it should be fast.
            break
            
        # Check all subsets of size k
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_triangle_free(mask):
                return k
                
    return 1

def _find_large_tf_set(adj_mask, n, target_size=15):
    """
    Find a triangle-free set of size target_size.
    Returns the set as a bitmask or None.
    """
    for subset in itertools.combinations(range(n), target_size):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        # Check if triangle free
        v_bits = mask
        is_tf = True
        while v_bits:
            lb = v_bits & (-v_bits)
            u = lb.bit_length() - 1
            v_bits ^= lb
            N_u = adj_mask[u] & mask
            w_bits = N_u
            while w_bits:
                lb2 = w_bits & (-w_bits)
                w = lb2.bit_length() - 1
                w_bits ^= lb2
                if adj_mask[w] & N_u:
                    is_tf = False
                    break
            if not is_tf:
                break
        if is_tf:
            return mask
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    n = 18
    all_edges = list(itertools.combinations(range(n), 2))
    
    def time_ok():
        return time.time() < deadline

    best_score = 18
    best_adj_mask = [0] * n
    
    def evaluate(adj_mask):
        if not _is_k4_free_mask(adj_mask, n):
            return None
        # We only care if score < best_score
        # If best_score is 15, we need to check if it's 14.
        # Checking for 15 is fast (C(18,15)=816).
        # Checking for 14 is C(18,14)=3060.
        # Checking for 13 is C(18,13)=8568.
        
        # Let's determine the exact max TF size.
        score = _max_tf_size_exact(adj_mask, n, timeout=1.0)
        return score

    # 1. Initial Construction: Random K4-free graph
    random.seed(42)
    
    # Generate a few random base graphs
    base_graphs = []
    for i in range(20):
        if not time_ok(): break
        adj = [0] * n
        edges = all_edges[:]
        random.shuffle(edges)
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not _is_k4_free_after_add(adj, n, u, v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        base_graphs.append(adj)
        
    # Evaluate base graphs
    for adj in base_graphs:
        if not time_ok(): break
        score = evaluate(adj)
        if score is not None and score < best_score:
            best_score = score
            best_adj_mask = adj[:]

    # 2. Directed Local Search
    # We want to reduce the max TF set size.
    # Strategy:
    # a. Find a large TF set S (e.g., size 15 or current best).
    # b. Try to add edges within S to break it.
    # c. Ensure no K4 is formed.
    
    current_adj = best_adj_mask[:]
    current_score = best_score
    
    for iteration in range(500):
        if not time_ok(): break
        
        # If we already have a very good score, we can try to improve further
        # If current_score is 15, we look for 14.
        # If current_score is 14, we look for 13.
        
        # Find a TF set of size current_score
        # If we can't find one, maybe current_score is overestimated? 
        # But _max_tf_size_exact is exact.
        
        # Let's try to find a TF set of size current_score.
        # If we find one, try to break it.
        
        tf_set_mask = _find_large_tf_set(current_adj, n, current_score)
        
        if tf_set_mask is None:
            # This shouldn't happen if score is exact, unless there are multiple sets and we picked a bad one?
            # Or maybe the score dropped due to previous edits?
            # Re-evaluate
            new_score = evaluate(current_adj)
            if new_score is not None and new_score < current_score:
                current_score = new_score
                best_score = new_score
                best_adj_mask = current_adj[:]
            continue
        
        # Get list of vertices in tf_set
        verts = []
        m = tf_set_mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            verts.append(v)
            m ^= lb
        
        # Try to add an edge between two non-adjacent vertices in this set
        # to create a triangle within the set.
        # For edge (u,v) in S, it creates a triangle if there is w in S adjacent to both u and v.
        
        candidates = []
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                u, v = verts[i], verts[j]
                if (current_adj[u] >> v) & 1:
                    continue # Already adjacent
                
                # Check if adding (u,v) creates a triangle in S
                # Common neighbors of u and v in S
                common_in_s = current_adj[u] & current_adj[v] & tf_set_mask
                if common_in_s:
                    # There is a w in S adjacent to both. Adding (u,v) creates triangle (u,v,w).
                    # This breaks the TF property of S.
                    candidates.append((u, v))
        
        if not candidates:
            # No single edge addition can break this specific set S.
            # This implies S is an independent set or has no "almost triangles".
            # If S is independent, we need to add edges, but they won't create triangles immediately.
            # We might need to add multiple edges.
            # For now, let's try a different TF set or restart local search from a different base.
            
            # Try to find another TF set of the same size
            # Or just break and try a new random base
            break
        
        # Pick a candidate randomly
        u, v = random.choice(candidates)
        
        # Check if adding (u,v) keeps graph K4-free
        # We need to check K4s containing (u,v).
        # This is done by _is_k4_free_after_add
        # But we also need to ensure we don't create a K4 with vertices OUTSIDE S?
        # _is_k4_free_after_add checks ALL common neighbors, not just in S.
        
        if _is_k4_free_after_add(current_adj, n, u, v):
            # Add edge
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            
            # Evaluate new score
            new_score = evaluate(current_adj)
            if new_score is not None:
                if new_score < current_score:
                    current_score = new_score
                    best_score = new_score
                    best_adj_mask = current_adj[:]
                # If score didn't improve, we might still keep the edge if it helps future steps?
                # But if score stays same, we might be stuck.
                # Let's keep it for now, as it might break other large TF sets.
            else:
                # Should not happen if _is_k4_free_after_add is correct
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
        else:
            # Cannot add this edge
            pass
            
    # 3. Final Polish: Try all single edge additions to best graph
    if time_ok():
        adj = best_adj_mask[:]
        improved = True
        while improved and time_ok():
            improved = False
            for u, v in all_edges:
                if not time_ok(): break
                if (adj[u] >> v) & 1:
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if _is_k4_free_after_add(adj, n, u, v):
                    score = evaluate(adj)
                    if score is not None and score < best_score:
                        best_score = score
                        best_adj_mask = adj[:]
                        improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

    # Convert to edge list
    best_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj_mask[u] >> v) & 1:
                best_edges.append([u, v])
    
    return best_edges
