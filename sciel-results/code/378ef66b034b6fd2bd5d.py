def greedy_alpha(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    remaining = set(range(n))
    alpha = 0
    while remaining:
        # Pick vertex with min degree in remaining
        min_deg = float('inf')
        min_v = -1
        for v in remaining:
            deg = len(adj[v] & remaining)
            if deg < min_deg:
                min_deg = deg
                min_v = v
        alpha += 1
        remaining.remove(min_v)
        remaining -= adj[min_v]
    return alpha
