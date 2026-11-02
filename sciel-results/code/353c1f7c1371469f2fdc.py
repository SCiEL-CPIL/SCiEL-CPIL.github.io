    def has_girth_ge(G, N, g):
        max_depth = g // 2
        for u in range(N):
            # BFS
            dist = [-1] * N
            parent = [-1] * N
            queue = [(u, 0)]
            dist[u] = 0
            while queue:
                curr, d = queue.pop(0)
                if d >= max_depth:
                    continue
                for v in G[curr]:
                    if v == parent[curr]:
                        continue
                    if dist[v] == -1:
                        dist[v] = d + 1
                        parent[v] = curr
                        queue.append((v, d + 1))
                    else:
                        # Found a cycle
                        # The cycle length is d + 1 + dist[v]?
                        # Actually, if we see a visited node v (not parent), 
                        # the cycle length is d + 1 + (distance from u to v in tree)?
                        # No, the path from u to curr is d. Edge curr-v.
                        # Path from u to v is dist[v].
                        # Cycle length is d + 1 + dist[v]?
                        # Wait, if v is visited, it's in the tree.
                        # The path u->...->v has length dist[v].
                        # The path u->...->curr->v has length d+1.
                        # The cycle is formed by these two paths.
                        # Length is dist[v] + (d+1) - 2 * LCA?
                        # This is getting complicated.
                        # Simpler: If we find ANY back-edge or cross-edge in BFS, 
                        # does it guarantee a cycle shorter than g?
                        # If max_depth = g/2, and we find a connection between two nodes 
                        # in the BFS tree of depth <= g/2, the cycle length is <= g/2 + g/2 = g.
                        # If the cycle length is exactly g, is it allowed?
                        # "no cycle shorter than g". So cycle length >= g is OK.
                        # So if we find a cycle of length L, we need L >= g.
                        # If we find a connection in BFS depth d1, d2, cycle len is d1+d2+1?
                        # If d1+d2+1 < g, return False.
                        # We need to calculate the cycle length.
                        pass
    