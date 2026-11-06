def kuhn_matching(adj, n_left, n_right):
    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    
    def dfs(u, visited):
        for v in adj[u]:
            if v in visited:
                continue
            visited.add(v)
            if pair_v[v] == -1 or dfs(pair_v[v], visited):
                pair_u[u] = v
                pair_v[v] = u
                return True
        return False

    matching = 0
    for u in range(n_left):
        visited = set()
        if dfs(u, visited):
            matching += 1
            
    return pair_u, pair_v
