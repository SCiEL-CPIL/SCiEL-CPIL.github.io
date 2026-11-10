import random

def construct():
    n = 20
    best_edges = 0
    best_graph = []
    
    # Theoretical max is 48.
    # We run random greedy restarts to find a graph with as many edges as possible.
    
    for _ in range(5000):
        adj = [set() for _ in range(n)]
        edges = []
        
        # Randomize edge order
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            # Check 1: Common neighbors of u and v must be <= 2
            if len(adj[u] & adj[v]) > 2:
                continue
            
            # Check 2: For all x in N(v), common neighbors of u and x must be <= 1
            # If we add (u, v), v becomes a neighbor of u.
            # For any x in N(v), v is in N(x). So v is in N(u) new ∩ N(x).
            # So |N(u) new ∩ N(x)| = |N(u) old ∩ N(x)| + 1 (since v not in N(u) old).
            # We need this <= 2, so |N(u) old ∩ N(x)| <= 1.
            valid = True
            for x in adj[v]:
                if len(adj[u] & adj[x]) > 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check 3: For all y in N(u), common neighbors of v and y must be <= 1
            for y in adj[u]:
                if len(adj[v] & adj[y]) > 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
            
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_graph = edges[:]
            
            if best_edges >= 48:
                break
                
    return best_graph
