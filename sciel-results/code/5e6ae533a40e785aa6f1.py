    def find_any_cycle_dfs():
        visited = [False] * n
        parent = [-1] * n
        cycle_found = []
        
        def dfs(u, p):
            visited[u] = True
            for v in adj[u]:
                if v == p:
                    continue
                if not visited[v]:
                    parent[v] = u
                    if dfs(v, u):
                        return True
                else:
                    # Back edge
                    # Reconstruct cycle
                    # u -> v is the back edge.
                    # Path is v -> ... -> u -> v
                    cycle = [v]
                    curr = u
                    while curr != v:
                        cycle.append(curr)
                        curr = parent[curr]
                    cycle.reverse()
                    # cycle is v, ..., u
                    # The cycle is v-...-u-v
                    cycle_found.extend(cycle)
                    return True
            return False
            
        for i in range(n):
            if not visited[i]:
                if dfs(i, -1):
                    return cycle_found
        return None
