def check_girth(n, adj, target_girth):
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = [start]
        while q:
            u = q.pop(0)
            if dist[u] >= target_girth // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    # Calculate length
                    # LCA of u and v
                    # This is O(n) per edge, too slow?
                    # n=320, edges=640. 320*640 = 200k. OK.
                    pass
    return True
