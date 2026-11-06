import time
import random
from itertools import combinations

def construct():
    n = 20
    time_limit = 110.0
    start_time = time.time()
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m = len(all_edges)
    
    # For K2,3-free: no two vertices can have 3+ common neighbors
    # Equivalently: for any 3 vertices x,y,z, they cannot all be adjacent to both u and v
    # i.e., if u and v both connect to x, y, z, then K2,3 is formed
    
    # Key insight: A set of edges forms a K2,3 if there exist vertices u,v 
    # such that at least 3 common neighbors exist.
    # 
    # Conflict: Two edges (u,x) and (v,x) don't conflict by themselves.
    # The conflict is: if we have edges (u,a), (u,b), (u,c) and (v,a), (v,b), (v,c),
    # then we have K2,3 with parts {u,v} and {a,b,c}.
    
    # So the "conflict graph" approach: each potential edge is a node.
    # Two edges conflict if... actually this is a 3-uniform hypergraph constraint.
    
    # Let me think differently. The constraint is:
    # For every pair of vertices (u,v), |N(u) ∩ N(v)| ≤ 2
    
    # This is a codegree constraint. Let's use a greedy + local search approach
    # that's more efficient than the current one.
    
    # Better approach: Start with a known good construction.
    # For n=20, we can try to build a graph with high density while maintaining codegree ≤ 2.
    
    # A Petersen-like or strongly regular graph might work.
    # SRG(20, k, λ, μ) where λ ≤ 2 and μ ≤ 2 would be K2,3-free.
    # 
    # Actually, let me just do a smarter greedy + local search.
    
    def codegree_ok(u, v, adj):
        """Check if adding edge (u,v) keeps codegree ≤ 2 for all pairs."""
        # Adding (u,v) means:
        # - For each w in N(v): codegree(u,w) increases by 1 (v becomes common neighbor)
        #   Need codegree(u,w) ≤ 1 before (so ≤ 2 after)
        # - For each w in N(u): codegree(v,w) increases by 1
        #   Need codegree(v,w) ≤ 1 before
        # - Pair (u,v): their common neighbors unchanged (u not in N(u), v not in N(v))
        
        # Check pair (u, w) for w in N(v):
        w = adj[v]
        while w:
            lsb = w & -w
            w ^= lsb
            idx = lsb.bit_length() - 1
            if (adj[u] & adj[idx]).bit_count() > 1:
                return False
        
        # Check pair (v, w) for w in N(u):
        w = adj[u]
        while w:
            lsb = w & -w
            w ^= lsb
            idx = lsb.bit_length() - 1
            if (adj[v] & adj[idx]).bit_count() > 1:
                return False
        
        return True
    
    def greedy_build(order, rng):
        adj = [0] * n
        edges = []
        for u, v in order:
            if codegree_ok(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        return edges, adj
    
    def local_search_improved(edges, adj, rng):
        """Local search: try adding missing edges, then try 1-for-2 swaps."""
        current_edges = set(edges)
        current_adj = adj[:]
        
        improved = True
        iterations = 0
        max_iterations = 50
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Phase 1: Try to add any single missing edge
            missing = [e for e in all_edges if e not in current_edges]
            rng.shuffle(missing)
            
            for u, v in missing:
                if codegree_ok(u, v, current_adj):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    improved = True
                    break
            
            # Phase 2: If no single edge can be added, try swaps
            if not improved:
                # Try removing one edge and adding one or more others
                edge_list = list(current_edges)
                rng.shuffle(edge_list)
                
                swap_found = False
                for u, v in edge_list:
                    # Remove (u,v)
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
                    current_edges.discard((u, v))
                    
                    # Try to add as many missing edges as possible
                    missing_after = [e for e in all_edges if e not in current_edges]
                    rng.shuffle(missing_after)
                    
                    added_count = 0
                    added_edges = []
                    for cu, cv in missing_after:
                        if codegree_ok(cu, cv, current_adj):
                            current_adj[cu] |= (1 << cv)
                            current_adj[cv] |= (1 << cu)
                            current_edges.add((cu, cv))
                            added_edges.append((cu, cv))
                            added_count += 1
                        if time.time() - start_time > time_limit - 5:
                            break
                    
                    # Check if we gained edges
                    if added_count >= 2:  # Removed 1, added 2+ => net gain
                        swap_found = True
                        break
                    else:
                        # Revert
                        for cu, cv in added_edges:
                            current_adj[cu] ^= (1 << cv)
                            current_adj[cv] ^= (1 << cu)
                            current_edges.discard((cu, cv))
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                        current_edges.add((u, v))
                
                if swap_found:
                    improved = True
                else:
                    break
        
        return list(current_edges)
    
    best_edges = []
    best_count = 0
    rng = random.Random(42)
    
    # Multiple random restarts
    num_restarts = 300
    for i in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Random order with some structure
        order = all_edges[:]
        rng.shuffle(order)
        
        edges, adj = greedy_build(order, rng)
        edges = local_search_improved(edges, adj, rng)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
        
        if best_count >= 160:  # Good enough, stop early
            break
    
    # Verification
    def is_valid(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] & adj[v]).bit_count() >= 3:
                    return False
        return True
    
    if not is_valid(best_edges) or len(best_edges) == 0:
        # Fallback: cycle
        return [[i, (i + 1) % 20] for i in range(20)]
    
    return [[u, v] for u, v in best_edges]
