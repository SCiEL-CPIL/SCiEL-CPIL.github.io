import random
import time
import sys

def construct():
    n = 20
    deadline = time.time() + 110  # Leave 10s for cleanup/exit
    
    # State: adjacency matrix or set of edges
    # We'll use a set of edges for easy add/remove and iteration
    # And a helper to check validity quickly
    
    def is_valid(edges):
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check common neighbors for all pairs
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u].intersection(adj[v])
                if len(common) >= 3:
                    return False
        return True

    def try_add_edge(edges, adj, u, v):
        # Check if adding (u, v) creates a K2,3
        # 1. Do u and v have >= 3 common neighbors?
        if len(adj[u].intersection(adj[v])) >= 3:
            return False
        
        # 2. For any w in adj[v], does adj[u] intersect adj[w] have size >= 2?
        #    Because if we add (u,v), v becomes a common neighbor of u and w.
        #    So current common neighbors of u,w must be < 2.
        for w in adj[v]:
            if w == u: continue
            if len(adj[u].intersection(adj[w])) >= 2:
                return False
                
        # 3. For any z in adj[u], does adj[v] intersect adj[z] have size >= 2?
        for z in adj[u]:
            if z == v: continue
            if len(adj[v].intersection(adj[z])) >= 2:
                return False
                
        return True

    def add_edge(edges, adj, u, v):
        edges.add((min(u, v), max(u, v)))
        adj[u].add(v)
        adj[v].add(u)

    def remove_edge(edges, adj, u, v):
        edges.discard((min(u, v), max(u, v)))
        adj[u].discard(v)
        adj[v].discard(u)

    best_edges = 0
    best_graph_edges = set()
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    while time.time() < deadline:
        # Start with a random graph
        # To get a good density, start with p=0.6 or similar
        # Or start with a known good structure and perturb
        # Let's try random restart with random initial density
        
        # Initialize
        edges = set()
        adj = {i: set() for i in range(n)}
        
        # Random initialization
        # We want to end up with high degree, so start dense
        for e in all_edges:
            if random.random() < 0.6:
                edges.add(e)
                u, v = e
                adj[u].add(v)
                adj[v].add(u)
                
        # Repair phase: remove edges until valid
        # To speed up, we can check validity globally or locally.
        # Global check is O(n^2 * deg). n=20, so 400 * 10 = 4000 ops. Very fast.
        
        valid = False
        repair_iters = 0
        while not valid and repair_iters < 1000:
            valid = is_valid(edges)
            if valid:
                break
            
            # Find a violation
            # We need to find a pair (u,v) with >= 3 common neighbors
            # And remove an edge to fix it.
            # To be efficient, we can just pick a random edge and remove it?
            # No, that might take too long.
            
            # Find one violating pair
            viol_u, viol_v = None, None
            for u in range(n):
                for v in range(u + 1, n):
                    if len(adj[u].intersection(adj[v])) >= 3:
                        viol_u, viol_v = u, v
                        break
                if viol_u is not None:
                    break
            
            if viol_u is None:
                valid = True
                break
            
            # Remove a random edge incident to viol_u or viol_v
            # This is a heuristic. Removing an edge incident to one of the vertices
            # reduces the degree and potentially the common neighbor counts.
            # Better: Remove an edge that is part of the "triangle" of common neighbors?
            # If u, v have common neighbors c1, c2, c3.
            # Edges (u, c1), (v, c1) etc are involved.
            # Removing (u, c1) will reduce common neighbors of (u, c2)? No.
            # Removing (u, c1) reduces common neighbors of (u, v) by 1 (if c1 was common).
            # It also affects (u, w) for w in N(c1).
            
            # Simple strategy: Remove a random edge incident to viol_u or viol_v
            candidates = list(adj[viol_u]) + list(adj[viol_v])
            if candidates:
                target = random.choice(candidates)
                remove_edge(edges, adj, viol_u, target) if random.random() < 0.5 else remove_edge(edges, adj, viol_v, target)
            else:
                # Should not happen if there is a violation
                break
                
            repair_iters += 1
            
        if not valid:
            continue
            
        # Improve phase
        # Try to add random missing edges
        # Shuffle missing edges
        missing = [e for e in all_edges if e not in edges]
        random.shuffle(missing)
        
        for u, v in missing:
            if try_add_edge(edges, adj, u, v):
                add_edge(edges, adj, u, v)
                
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_graph_edges = set(edges)
            
    # Return list of edges
    return [list(e) for e in best_graph_edges]
