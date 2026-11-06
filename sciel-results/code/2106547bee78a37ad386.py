import random
import itertools

def construct():
    n = 20
    random.seed(42)
    
    # Start with a moderately dense random graph
    p = 0.45
    adj = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                adj[i][j] = adj[j][i] = True
    
    def is_k33_free(adj):
        # Check no three vertices have 3 common neighbors
        for triple in itertools.combinations(range(n), 3):
            u, v, w = triple
            common = [x for x in range(n) if adj[u][x] and adj[v][x] and adj[w][x]]
            if len(common) >= 3:
                return False, triple, common
        return True, None, None
    
    # Find and remove edges from K3,3 subgraphs
    max_iters = 5000
    for _ in range(max_iters):
        ok, triple, common = is_k33_free(adj)
        if ok:
            break
        # Remove a random edge from the K3,3 (the K3,3 has edges between triple and common)
        # Actually, the K3,3 is formed by triple and any 3 from common
        # Remove a random edge between triple and common
        t = random.choice(triple)
        c = random.choice(common)
        if adj[t][c]:
            adj[t][c] = adj[c][t] = False
    
    # Try to add edges back
    added = True
    while added:
        added = False
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    adj[i][j] = adj[j][i] = True
                    ok, _, _ = is_k33_free(adj)
                    if not ok:
                        adj[i][j] = adj[j][i] = False
                    else:
                        added = True
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges
