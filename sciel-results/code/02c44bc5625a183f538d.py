# EVOLVE-BLOCK-START
import time
import itertools

def _is_valid(edges, n=15):
    """Check if no 3-edge path s-u-v-t has d(s)==d(t)."""
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    deg = [sum(adj[i]) for i in range(n)]
    # For each edge (u,v), check all s in N(u)\{v} and t in N(v)\{u}, s!=t
    for u, v in edges:
        for s in range(n):
            if s == u or s == v:
                continue
            if not adj[u][s]:
                continue
            for t in range(n):
                if t == u or t == v or t == s:
                    continue
                if not adj[v][t]:
                    continue
                if deg[s] == deg[t]:
                    return False
    return True

def _try_add_edges(base_edges, n=15, deadline=None):
    """Try to add edges to base_edges while maintaining validity."""
    edges = list(base_edges)
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Generate all possible additional edges not already present
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if not adj[u][v]:
                candidates.append((u, v))
    
    # Greedy: try adding each candidate if valid
    added = True
    while added:
        added = False
        if deadline and time.time() > deadline:
            break
        for (u, v) in candidates:
            if deadline and time.time() > deadline:
                break
            if adj[u][v]:
                continue
            # Try adding edge (u,v)
            adj[u][v] = True
            adj[v][u] = True
            test_edges = edges + [[u, v]]
            if _is_valid(test_edges, n):
                edges.append([u, v])
                added = True
            else:
                adj[u][v] = False
                adj[v][u] = False
    
    return edges

def construct():
    """
    Returns edges of K_{7,8} on vertices 0..14, then attempts to add more edges.
    Part A = {0..6}, Part B = {7..14}.
    Base: All A vertices have degree 8, all B vertices have degree 7.
    Edge count = 56.
    """
    n = 15
    A = list(range(0, 7))
    B = list(range(7, 15))
    
    base_edges = []
    for u in A:
        for v in B:
            base_edges.append([u, v])
    
    deadline = time.time() + 70  # safety margin
    
    # Try to add more edges greedily
    improved = _try_add_edges(base_edges, n, deadline)
    
    # Ensure exact types: Python int for vertices, list of lists
    result = [[int(u), int(v)] for u, v in improved]
    return result
# EVOLVE-BLOCK-END
