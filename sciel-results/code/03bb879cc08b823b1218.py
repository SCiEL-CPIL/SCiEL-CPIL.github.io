    def check_girth(n, adj, g):
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                du = dist[u]
                if du >= g - 1:
                    continue
                for v in adj[u]:
                    if v == s:
                        if du + 1 < g:
                            return False
                    elif dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # v is visited
                        if v == parent[u]:
                            continue
                        # Cycle length is du + dist[v] + 1
                        # This is an upper bound on the actual cycle length?
                        # No, if they share a prefix, the cycle is shorter.
                        # So if du + dist[v] + 1 < g, then actual cycle < g.
                        if du + dist[v] + 1 < g:
                            return False
        return True
