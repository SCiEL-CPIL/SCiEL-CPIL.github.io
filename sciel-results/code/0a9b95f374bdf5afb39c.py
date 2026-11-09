"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_triangle_free(adj, n, u, v):
    """Check if adding edge (u,v) creates a triangle."""
    common = set(adj[u]) & set(adj[v])
    return len(common) == 0

def _is_not_3_colorable(adj, n):
    """Check if graph is NOT 3-colorable using backtracking."""
    def backtrack(coloring, vertex):
        if vertex == n:
            return True
        for c in range(3):
            if all(coloring[neighbor] != c for neighbor in adj[vertex]):
                coloring[vertex] = c
                if backtrack(coloring, vertex + 1):
                    return True
                coloring[vertex] = -1
        return False
    
    # Try to 3-color the graph
    coloring = [-1] * n
    return not backtrack(coloring, 0)

def construct():
    deadline = time.time() + 140  # 140s safety margin
    
    # Start with Groetzsch graph on vertices 0..10
    n = 16
    adj = [set() for _ in range(n)]
    
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    for u, v in cycle:
        adj[u].add(v)
        adj[v].add(u)
    for u, w in cycle:
        adj[u].add(5 + w)
        adj[5 + w].add(u)
        adj[w].add(5 + u)
        adj[5 + u].add(w)
    for u in range(5):
        adj[5 + u].add(10)
        adj[10].add(5 + u)
    
    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges
    
    best_edges = get_edges(adj)
    best_adj = [set(s) for s in adj]
    
    # Local search: try to add edges
    random.seed(42)
    for iteration in range(100000):
        if time.time() > deadline:
            break
        
        # Try adding a random missing edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v or v in adj[u]:
            continue
        
        # Check triangle-freeness
        if not _is_triangle_free(adj, n, u, v):
            continue
        
        # Try adding the edge
        adj[u].add(v)
        adj[v].add(u)
        
        # Check if still not 3-colorable
        if _is_not_3_colorable(adj, n):
            # Keep the edge, update best if better
            current_edges = get_edges(adj)
            if len(current_edges) > len(best_edges):
                best_edges = current_edges
                best_adj = [set(s) for s in adj]
        else:
            # Remove the edge
            adj[u].remove(v)
            adj[v].remove(u)
    
    return best_edges
# EVOLVE-BLOCK-END
