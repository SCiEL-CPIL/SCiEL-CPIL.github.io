import time
import random

def construct():
    """
    Constructs a dense graph on 15 vertices avoiding equal-degree 3-path endpoints.
    Uses local search to maximize edges while maintaining the constraint.
    """
    n = 15
    deadline = time.time() + 85  # safety margin before 90s limit
    
    def is_valid(adj):
        """Check if no 3-path s-u-v-t has deg(s) == deg(t)."""
        deg = [sum(adj[i]) for i in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                # For each edge (u,v), check all 3-paths s-u-v-t
                for s in range(n):
                    if s == u or s == v:
                        continue
                    if not adj[s][u]:
                        continue
                    for t in range(n):
                        if t == u or t == v or t == s:
                            continue
                        if not adj[v][t]:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2
    
    # Start from K_{7,8}
    adj = [[0]*n for _ in range(n)]
    for u in range(7):
        for v in range(7, n):
            adj[u][v] = 1
            adj[v][u] = 1
    
    best_adj = [row[:] for row in adj]
    best_edges = count_edges(adj)
    
    # Local search: try adding edges
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    continue
                # Try adding edge (u,v)
                adj[u][v] = 1
                adj[v][u] = 1
                if is_valid(adj):
                    new_edges = count_edges(adj)
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = [row[:] for row in adj]
                    changed = True
                else:
                    adj[u][v] = 0
                    adj[v][u] = 0
    
    # Try removing some edges to allow adding more
    if time.time() < deadline:
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                if time.time() > deadline:
                    break
                # Try removing edge (u,v)
                adj[u][v] = 0
                adj[v][u] = 0
                # Now try to add other edges
                for a in range(n):
                    for b in range(a+1, n):
                        if adj[a][b]:
                            continue
                        adj[a][b] = 1
                        adj[b][a] = 1
                        if is_valid(adj):
                            new_edges = count_edges(adj)
                            if new_edges > best_edges:
                                best_edges = new_edges
                                best_adj = [row[:] for row in adj]
                        else:
                            adj[a][b] = 0
                            adj[b][a] = 0
                # Restore
                adj[u][v] = 1
                adj[v][u] = 1
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                edges.append([u, v])
    
    return edges
