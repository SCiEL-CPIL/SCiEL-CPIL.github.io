"""Program for dense odd-order graphs avoiding equal-degree path endpoints."""

# EVOLVE-BLOCK-START
P = {'n': 15}

def construct():
    n = P['n']
    
    # Start with star: vertex 0 connected to all others
    adj = [[False] * n for _ in range(n)]
    for v in range(1, n):
        adj[0][v] = adj[v][0] = True
    
    # Compute current degrees
    def get_degrees():
        return [sum(adj[i]) for i in range(n)]
    
    def check_valid(adj, n):
        """Check if any simple 3-edge path s-u-v-t has deg(s)==deg(t)."""
        deg = get_degrees()
        # For each edge (u,v), check all s in N(u)\{v} and t in N(v)\{u}
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                for s in range(n):
                    if s == u or s == v:
                        continue
                    if not adj[s][u]:
                        continue
                    for t in range(n):
                        if t == s or t == u or t == v:
                            continue
                        if not adj[v][t]:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    # Greedily add edges
    added = True
    while added:
        added = False
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    continue
                # Try adding edge (i,j)
                adj[i][j] = adj[j][i] = True
                if check_valid(adj, n):
                    added = True
                    break
                else:
                    adj[i][j] = adj[j][i] = False
            if added:
                break
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    
    # Final validation
    assert check_valid(adj, n), "Graph is invalid!"
    
    # Convert to proper Python ints
    edges = [[int(u), int(v)] for u, v in edges]
    return edges
# EVOLVE-BLOCK-END
