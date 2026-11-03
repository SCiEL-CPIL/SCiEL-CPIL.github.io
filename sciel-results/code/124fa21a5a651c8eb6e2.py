<<<<<<< SEARCH
    def verify_girth(n, edges, target_girth):
        """Verify girth >= target_girth using BFS. Returns actual girth."""
        if n == 0:
            return 999
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_girth = 999
        
        for start in range(n):
            if time.time() > deadline:
                return min_girth
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                cur = queue.popleft()
                for nb in adj[cur]:
                    if dist[nb] == -1:
                        dist[nb] = dist[cur] + 1
                        parent[nb] = cur
                        queue.append(nb)
                    elif parent[nb] != cur and parent[cur] != nb:
                        cycle_len = dist[nb] + dist[cur] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if min_girth >= target_girth:
                                return min_girth
        
        return min_girth
=======
    def verify_girth(n, edges, target_girth):
        """Verify girth >= target_girth using BFS. Returns actual girth."""
        if n == 0:
            return 999
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_girth = 999
        
        # Limit verification time to 10s per graph to allow trying multiple depths
        local_deadline = time.time() + 10
        
        for start in range(n):
            if time.time() > local_deadline:
                return 0 # Timeout, assume invalid for now
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                cur = queue.popleft()
                for nb in adj[cur]:
                    if dist[nb] == -1:
                        dist[nb] = dist[cur] + 1
                        parent[nb] = cur
                        queue.append(nb)
                    elif parent[nb] != cur and parent[cur] != nb:
                        cycle_len = dist[nb] + dist[cur] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if min_girth >= target_girth:
                                return min_girth
        
        return min_girth
>>>>>>> REPLACE
