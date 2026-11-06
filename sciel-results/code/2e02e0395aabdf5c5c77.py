import time
import random

def _is_valid_graph(n, edges):
    """Check if graph has no K4 and no independent set of size 6."""
    # Build adjacency matrix as list of bitsets for speed
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Check for K4
    # A K4 exists if there is a triangle (u,v,w) such that all are connected.
    # Or simply: for every edge (u,v), check if they have a common neighbor w,
    # and if w is connected to some x... 
    # Efficient check: Iterate all triples? n=40, triples ~ 10k. 
    # For each triple, check if it's a triangle.
    # If triangle, check if any vertex in triangle is connected to all others in a 4th set?
    # Simpler: 
    # For each pair (u, v) that is an edge:
    #   common = adj[u] & adj[v]
    #   For each w in common:
    #     # u,v,w form a triangle.
    #     # Check if there is an x connected to u, v, w.
    #     # x must be in adj[u] & adj[v] & adj[w].
    #     # So if (adj[u] & adj[v] & adj[w]) != 0, we have a K4.
    #     # Note: x cannot be u, v, w. But adj doesn't include self-loops, so x is distinct.
    #     if (adj[u] & adj[v] & adj[w]) != 0:
    #         return False
    #     # Wait, if x is in common of u,v,w, then u,v,w,x form K4.
    #     # So if the intersection is non-empty, K4 exists.
    
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                # Iterate w in common
                # We only need to check if common has a bit that is also in adj[w]?
                # Actually, if w is in common, then u~w and v~w.
                # We need x such that x~u, x~v, x~w.
                # So x must be in adj[u] & adj[v] & adj[w].
                # So we just need to check if (adj[u] & adj[v] & adj[w]) is non-zero.
                # But w is in common, so w is in adj[u] and adj[v].
                # So adj[u] & adj[v] contains w.
                # So (adj[u] & adj[v] & adj[w]) will contain w?
                # No, adj[w] does not contain w.
                # So if there is ANY x in the intersection, it's a K4.
                # The intersection might contain w? No, w not in adj[w].
                # So if intersection is non-empty, it contains some x != u,v,w.
                # So K4 exists.
                
                # Optimization: 
                # If common has bits, pick one w.
                # Check if (adj[u] & adj[v] & adj[w]) != 0.
                # But we need to check ALL w?
                # If there is a K4, say {u,v,w,x}, then for edge (u,v), w is in common.
                # And x is in adj[u]&adj[v]&adj[w].
                # So yes, checking any w in common that leads to non-empty intersection works.
                # Actually, if {u,v,w,x} is a K4, then for any pair, the intersection of their neighbors contains the other two.
                # So we can just check:
                # For each edge (u,v):
                #   common = adj[u] & adj[v]
                #   If common is 0, continue.
                #   For each w in common:
                #      if (adj[u] & adj[v] & adj[w]) != 0: return False
                # This is O(E * deg^2). For n=40, deg~20, E~400. 400*400 = 160k ops. Fast.
                
                # Even faster:
                # If (adj[u] & adj[v]) has at least 2 bits, say w and x.
                # If w and x are connected, then u,v,w,x is a K4.
                # So we need to check if any two vertices in common are connected.
                # i.e., if common & adj[w] != 0 for some w in common?
                # Yes.
                
                # Let's just do the triple check.
                # To avoid double counting, we can just check if there exists w in common such that 
                # (adj[u] & adj[v] & adj[w]) != 0.
                # Actually, if u,v,w,x is a K4, then w and x are both in common(u,v).
                # And w is connected to x.
                # So if we pick w from common, and check if w is connected to ANY other vertex in common, we find it.
                # So:
                #   for w in bits(common):
                #      if (common & adj[w]) != 0: return False
                #      # This checks if w is connected to any other vertex in common.
                #      # If yes, that vertex x forms K4 with u,v,w.
                
                # This is O(deg^2) per edge.
                common = adj[u] & adj[v]
                # Iterate bits in common
                m = common
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    m ^= lsb
                    if common & adj[w]:
                        return False

    # Check for independent set of size 6
    # Use a simple backtracking or greedy + check.
    # Since n is small, we can use a standard MIS algorithm.
    # Or just check all subsets of size 6? C(40,6) ~ 3.8M.
    # 3.8M * 15 edge checks ~ 50M ops. Might be slow in Python (1-2s).
    # We can prune.
    
    # Better: Use a bitset-based MIS search.
    # Or, since we suspect alpha <= 5, we can try to find a set of size 6.
    
    # Let's use a simple recursive search with pruning.
    # Or, use the fact that if we can't find a clique of size 6 in the complement, we are safe.
    # Complement graph: u ~ v if NOT edge in G.
    # We need to check if Complement has a clique of size 6.
    
    # Build complement adjacency
    comp_adj = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = full_mask ^ adj[i] ^ (1 << i)
        
    # Check for clique of size 6 in comp_adj
    # Recursive search
    def has_clique_size(k, adj_list, n):
        # Use a simple backtracking
        # State: current set, start index
        # For n=40, k=6, this is fast if we prune well.
        
        # Greedy upper bound: 
        # If remaining vertices < k - current_size, return False.
        
        # We'll use a simple DFS.
        def dfs(verts, depth):
            if depth == k:
                return True
            if not verts:
                return False
            # Pruning: if len(verts) < k - depth: return False
            
            # Pick a vertex
            # To be efficient, pick vertex with highest degree in induced subgraph?
            # Or just iterate.
            
            # Bitset representation of verts
            v = verts
            while v:
                lsb = v & -v
                u = lsb.bit_length() - 1
                v ^= lsb
                # Next candidates must be neighbors of u in comp_adj, and in verts
                new_verts = v & comp_adj[u]
                if dfs(new_verts, depth + 1):
                    return True
            return False
            
        return dfs(full_mask, 0)

    if has_clique_size(6, comp_adj, n):
        return False
        
    return True

def _paley_graph(p):
    """Generate Paley graph for prime p."""
    squares = {x * x % p for x in range(1, p)}
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                edges.append([i, j])
    return {'n': p, 'edges': edges}

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    candidates = [17, 29, 37]
    
    best_graph = None
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        g = _paley_graph(p)
        # Validate
        if _is_valid_graph(g['n'], g['edges']):
            best_graph = g
            # If we found a larger one, we can continue to see if even larger works?
            # The list is sorted by size. 17, 29, 37.
            # If 29 works, 37 might not.
            # We want the MAX n.
            # So we should check 37 first?
            # But 37 is more likely to fail.
            # Let's check in descending order of n?
            # No, if 37 fails, we fall back to 29.
            # If 29 fails, we fall back to 17.
            
    # If none worked, return 17 (which is known valid)
    if best_graph is None:
        best_graph = _paley_graph(17)
        
    return best_graph
# EVOLVE-BLOCK-END
