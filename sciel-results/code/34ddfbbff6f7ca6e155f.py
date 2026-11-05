import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def is_valid_graph(adj_masks, n):
        """
        Checks if the graph has no regular induced subgraph of size >= 6.
        """
        # Check size 6
        for subset in itertools.combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            degs = []
            for v in subset:
                degs.append((adj_masks[v] & mask).bit_count())
            if len(set(degs)) == 1:
                return False
        
        # Check larger sizes (rare, but necessary for correctness)
        # For n=20, C(20,7) is 77520. C(20,8) is 125970.
        # This is expensive. 
        # However, if we construct via greedy, we only guarantee size 6 is clean.
        # We must verify larger sizes at the end.
        for k in range(7, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                degs = []
                for v in subset:
                    degs.append((adj_masks[v] & mask).bit_count())
                if len(set(degs)) == 1:
                    return False
        return True

    def check_size6_violation(adj_masks, k, prev_vertices, candidate_mask):
        """
        Checks if adding vertex k with edges defined by candidate_mask to prev_vertices
        creates any regular induced subgraph of size 6.
        candidate_mask is a bitmask of length k, where bit i is 1 if edge (k, i) exists.
        prev_vertices is [0, 1, ..., k-1].
        """
        # Any size-6 subgraph including k must consist of k and 5 vertices from prev_vertices.
        # We need to check all combinations of 5 vertices from prev_vertices.
        # For each such set S, let T be the neighbors of k in S.
        # deg(k) = |T|.
        # For v in S, deg(v) in the induced subgraph is:
        #   (deg of v in G_{k-1} restricted to S) + (1 if v in T else 0)
        
        # Precompute degrees of vertices in prev_vertices restricted to any subset?
        # No, just compute on the fly for each candidate S.
        
        # Optimization: 
        # deg(k) must be between 0 and 5.
        # Let d_k = bit_count(candidate_mask & mask_S).
        # For v in S, let d_v = (adj_masks[v] & mask_S).bit_count() + (1 if v in T else 0).
        # Note: adj_masks[v] currently only has edges to vertices < v (if we build incrementally) 
        # OR we maintain full adjacency masks for the partial graph.
        # Let's assume adj_masks[v] contains edges to all vertices in the current partial graph (0..k-1).
        
        for S in itertools.combinations(range(k), 5):
            mask_S = 0
            for v in S:
                mask_S |= (1 << v)
            
            # Determine neighbors of k in S
            T_mask = candidate_mask & mask_S
            d_k = T_mask.bit_count()
            
            # Check if all degrees in S U {k} are equal to d_k
            # If d_k is 0, all v in S must have degree 0 in the induced subgraph.
            # If d_k is 5, all v in S must have degree 5 (impossible as max deg in S is 4 + 1 = 5? 
            # Wait, max degree in S is 4 (edges within S). Plus edge to k. So max 5.
            
            is_regular = True
            for v in S:
                # Degree of v within S
                deg_in_S = (adj_masks[v] & mask_S).bit_count()
                # Plus edge to k if v is in T
                if T_mask & (1 << v):
                    deg_total = deg_in_S + 1
                else:
                    deg_total = deg_in_S
                
                if deg_total != d_k:
                    is_regular = False
                    break
            
            if is_regular:
                return True # Violation found
        
        return False # No violation

    def greedy_search(n, max_restarts=1000):
        """
        Attempts to build a valid graph of size n using greedy backtracking.
        """
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                return None
            
            adj = [0] * n
            # adj[v] will contain edges to vertices < v initially? 
            # No, let's just store the full mask as we go.
            # But for the check, we only care about edges among 0..k-1.
            
            valid = True
            for k in range(n):
                if k < 5:
                    # No size-6 subgraphs possible yet.
                    # We can just add random edges to previous vertices.
                    for i in range(k):
                        if random.random() < 0.5:
                            adj[k] |= (1 << i)
                            adj[i] |= (1 << k)
                    continue
                
                # k >= 5
                # We need to choose edges from k to 0..k-1.
                # Try random masks until we find one that doesn't violate size-6 constraints.
                found_valid_mask = False
                attempts = 0
                max_attempts = 500 # Limit attempts per vertex
                
                while not found_valid_mask and attempts < max_attempts:
                    attempts += 1
                    # Generate random mask for edges from k to 0..k-1
                    # To avoid bias, we can pick density ~ 0.5
                    mask = 0
                    for i in range(k):
                        if random.random() < 0.5:
                            mask |= (1 << i)
                    
                    # Check if this mask creates any size-6 regular subgraph
                    if not check_size6_violation(adj, k, range(k), mask):
                        # Valid mask found
                        for i in range(k):
                            if mask & (1 << i):
                                adj[k] |= (1 << i)
                                adj[i] |= (1 << k)
                        found_valid_mask = True
                
                if not found_valid_mask:
                    # Backtrack or Restart
                    valid = False
                    break
            
            if valid:
                # We have a graph where NO size-6 subgraph is regular.
                # Now we MUST check for size >= 7.
                # This is the expensive part.
                if is_valid_graph(adj, n):
                    return adj
                else:
                    # It failed larger sizes. 
                    # We could try to repair, but for now, restart.
                    pass
        
        return None

    # Strategy:
    # 1. Check known small valid graphs?
    # 2. Try Greedy Search for n=21, 20.
    # 3. Fallback to Paley(19) if n=19 is needed.
    
    # First, let's verify if Paley(19) is valid and use it as a guaranteed n=19.
    def build_paley(n):
        if n != 19: return None
        p = 19
        adj = [0]*p
        for i in range(p):
            for j in range(i+1, p):
                if pow((i-j) % p, (p-1)//2, p) == 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    adj19 = build_paley(19)
    if adj19 and is_valid_graph(adj19, 19):
        # Keep this as fallback
        fallback_adj = adj19
    else:
        fallback_adj = None

    best_adj = None
    best_n = 0

    # Try to find larger graphs
    for target_n in [21, 20]:
        if time.time() > deadline - 10.0:
            break
        
        # Run greedy search
        adj = greedy_search(target_n, max_restarts=200)
        if adj:
            best_adj = adj
            best_n = target_n
            break # Found the largest possible (since we try 21 first)
    
    if best_adj:
        edges = []
        for i in range(best_n):
            m = best_adj[i]
            for j in range(i+1, best_n):
                if m & (1 << j):
                    edges.append([i, j])
        return {'n': best_n, 'edges': edges}
    
    # Fallback to n=19
    if fallback_adj:
        edges = []
        for i in range(19):
            m = fallback_adj[i]
            for j in range(i+1, 19):
                if m & (1 << j):
                    edges.append([i, j])
        return {'n': 19, 'edges': edges}
    
    # Fallback to n=5 (trivial)
    return {'n': 5, 'edges': []}
