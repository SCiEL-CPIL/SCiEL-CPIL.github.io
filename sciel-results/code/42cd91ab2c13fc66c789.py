    if parent[u] != v:
        cycle_len = d + dist[v] + 1
        if cycle_len < min_girth:
            return False
    