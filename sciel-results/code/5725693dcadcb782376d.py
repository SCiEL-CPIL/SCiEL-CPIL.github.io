import time
import random

def construct():
    n = 20
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    # Build all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m = len(all_edges)  # 190
    
    # Build conflict graph: two edges conflict if they cannot coexist in a K2,3-free graph
    # Edge i = (a,b), edge j = (c,d) conflict if:
    #   - They share a vertex AND the other endpoints have >= 2 common neighbors with the shared vertex
    #   - More precisely: adding both creates a K2,3 somewhere
    #
    # Actually, the constraint is: no two vertices have 3 common neighbors.
    # So edges (u,v) and (w,x) conflict if:
    #   - {u,v} and {w,x} share exactly one vertex, say u, and v,w have a common neighbor
    #     that together with some other structure creates K2,3
    #
    # Let me think more carefully. The K2,3 constraint: no two vertices have 3 common neighbors.
    # So for the graph to be K2,3-free, for every pair of vertices (p,q), |N(p) ∩ N(q)| <= 2.
    #
    # Two edges e1=(a,b) and e2=(c,d) are incompatible if adding both to some set would
    # cause some pair to have 3 common neighbors. This is complex to precompute.
    #
    # Alternative: use the same validation as the current program but in a B&B framework.
    # The conflict graph is implicit - I'll use the validate function during branching.
    
    # Actually, let me build an explicit conflict graph for better B&B performance.
    # Two edges (u,v) and (w,x) conflict if they cannot both be in a K2,3-free graph.
    # This happens when:
    # Case 1: They share a vertex. WLOG u=w. Then v and x are both neighbors of u.
    #   For any common neighbor c of v and x (other than u), if |N(v) ∩ N(x)| >= 2 
    #   (including u), then v and x have >= 2 common neighbors. But we need 3 for K2,3.
    #   Actually: if v and x have 2 common neighbors besides u, plus u = 3 common neighbors.
    #   So (u,v) and (u,x) conflict if v and x have >= 2 common neighbors (not counting u).
    #
    # Case 2: They don't share a vertex. e1=(a,b), e2=(c,d) with all distinct.
    #   Adding both: a-b and c-d. K2,3 requires two vertices with 3 common neighbors.
    #   The pairs to check are (a,c), (a,d), (b,c), (b,d).
    #   For pair (a,c): common neighbors include b (if a-b exists) and d (if c-d exists) 
    #   only if b=d or something... Actually b is neighbor of a, d is neighbor of c.
    #   b is common neighbor of a and c only if b is also neighbor of c.
    #   This is getting complex. Let me just use the validation function.
    
    # For B&B, I'll use a different approach:
    # - Represent the set of chosen edges as a bitmask
    # - Use the validation function to check if an edge can be added
    # - Branch on edges in a smart order
    
    # Precompute adjacency for quick validation
    # Use the same fast validation as current program
    
    def validate_and_add(adj, u, v):
        """Check if edge (u,v) can be added given current adj."""
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
        mask = adj[v]
        while mask:
            b = (mask & -mask).bit_length() - 1
            if (adj[u] & adj[b]).bit_count() > 1:
                return False
            mask &= mask - 1
        mask = adj[u]
        while mask:
            a = (mask & -mask).bit_length() - 1
            if (adj[v] & adj[a]).bit_count() > 1:
                return False
            mask &= mask - 1
        return True

    # Build explicit conflict graph for B&B
    # conflict[i] is a bitmask of edges that conflict with edge i
    conflict = [0] * m
    
    for i in range(m):
        u1, v1 = all_edges[i]
        for j in range(i + 1, m):
            u2, v2 = all_edges[j]
            
            # Check if edges (u1,v1) and (u2,v2) can coexist in a K2,3-free graph
            # They conflict if there's no way to have both without creating K2,3
            # Simple check: do they share a vertex?
            shared = set([u1, v1]) & set([u2, v2])
            
            if len(shared) == 1:
                # Share one vertex, say u1 == u2
                # The other endpoints are v1 and v2 (or v1 and u2, etc.)
                if u1 == u2:
                    a, b = v1, v2
                elif u1 == v2:
                    a, b = v1, u2
                elif v1 == u2:
                    a, b = u1, v2
                else:
                    a, b = u1, u2
                
                # a and b are both neighbors of the shared vertex
                # They conflict if a and b have >= 2 common neighbors (besides the shared vertex)
                # In the empty graph, they have 0 common neighbors, so no conflict initially
                # But in a dense graph, they might. For the conflict graph, we need to be conservative.
                # Two edges sharing a vertex conflict if their other endpoints could have 2+ common neighbors.
                # In the worst case (complete graph), every pair of vertices is adjacent, so
                # a and b would have n-2 = 18 common neighbors. So they ALWAYS conflict if they share a vertex?
                # No wait - in the complete graph, yes, but we're building a sparse graph.
                # For the conflict graph in B&B, we need: can these two edges BOTH be in SOME K2,3-free graph?
                # If they share a vertex u, and the other endpoints are a,b:
                # We need |N(a) ∩ N(b)| <= 2 in the final graph.
                # In the final graph, u is a common neighbor of a and b (since u-a and u-b exist).
                # So we need at most 1 OTHER common neighbor of a and b.
                # This is always satisfiable (just don't add other edges between a,b's neighborhoods).
                # So edges sharing a vertex do NOT inherently conflict.
                pass
            else:
                # No shared vertices: (u1,v1) and (u2,v2) with all distinct
                # K2,3 could form if some pair has 3 common neighbors
                # Pairs to worry about: (u1,u2), (u1,v2), (v1,u2), (v1,v2)
                # For (u1,u2): common neighbors include v1 (if v1-u2) and v2 (if v1-u2)... 
                # Actually v1 is neighbor of u1, v2 is neighbor of u2.
                # v1 is common neighbor of u1 and u2 only if v1 is also neighbor of u2.
                # In the minimal case with just these two edges, no common neighbors.
                # So no inherent conflict.
                pass
    
    # Since the conflict graph is not simply determined by edge pairs alone (it depends on the full graph),
    # I'll use a hybrid approach: B&B with dynamic validation
    
    # Strategy: Use the current program's local search but with a smarter initial solution
    # and more aggressive polishing. Also add a B&B-like approach for small subproblems.
    
    best_edges = []
    best_count = 0
    
    # Helper: build adj from edge list
    def build_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    # Helper: try to add all possible edges (greedy completion)
    def greedy_complete(adj, edges):
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if (adj[u] >> v) & 1:
                    continue
                if validate_and_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
                    improved = True
        return adj, edges
    
    # Helper: local search with edge swaps
    def local_search(adj, edges, max_iter=200):
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Try adding missing edges
            for u, v in all_edges:
                if (adj[u] >> v) & 1:
                    continue
                if validate_and_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-for-2 swap
            for i in range(len(edges)):
                u, v = edges[i]
                old_adj = adj[:]
                old_edges = edges[:]
                
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges.pop(i)
                
                added = 0
                for a, b in all_edges:
                    if added >= 2:
                        break
                    if (adj[a] >> b) & 1:
                        continue
                    if validate_and_add(adj, a, b):
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                        edges.append((a, b))
                        added += 1
                
                if added < 2:
                    adj[:] = old_adj
                    edges[:] = old_edges
                else:
                    improved = True
                    break
        
        return adj, edges
    
    # Multiple restarts with different strategies
    while time.time() < deadline - 5:
        # Random greedy construction
        adj = [0] * n
        current_edges = []
        
        edge_order = all_edges[:]
        random.shuffle(edge_order)
        
        for u, v in edge_order:
            if validate_and_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append((u, v))
        
        # Greedy completion
        adj, current_edges = greedy_complete(adj, current_edges)
        
        # Local search
        adj, current_edges = local_search(adj, current_edges, max_iter=300)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
        
        # If we have a good solution, do intensive polish
        if best_count >= 150 and time.time() < deadline - 10:
            adj2 = build_adj(best_edges)
            adj2, polished = local_search(adj2, best_edges[:], max_iter=500)
            if len(polished) > best_count:
                best_count = len(polished)
                best_edges = polished[:]
    
    # Final intensive polish on the best solution
    if best_count > 0 and time.time() < deadline - 3:
        adj_best = build_adj(best_edges)
        
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            adj_best, polished = local_search(adj_best, best_edges[:], max_iter=1000)
            if len(polished) > best_count:
                best_count = len(polished)
                best_edges = polished[:]
                adj_best = build_adj(best_edges)
    
    return [[u, v] for u, v in best_edges]
