    def find_shortest_paths_bfs(u, v, banned, limit=50):
        """
        Find up to 'limit' shortest simple paths from u to v avoiding 'banned' vertices in interior.
        Returns list of paths sorted by length.
        """
        if u == v:
            return [[u]]
        
        # BFS to find shortest distance
        dist = {u: 0}
        parent = {u: None}
        q = deque([u])
        found_v = False
        while q:
            cur = q.popleft()
            if cur == v:
                found_v = True
                break
            for nb in adj[cur]:
                if nb in banned and nb != v:
                    continue
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    parent[nb] = cur
                    q.append(nb)
        
        if not found_v or v not in dist:
            return []
        
        shortest_len = dist[v]
        
        # Now DFS to find all paths of length shortest_len
        # We use a visited set for the current path
        paths = []
        
        def dfs(cur, path, visited):
            if len(paths) >= limit:
                return
            if cur == v:
                if len(path) - 1 == shortest_len:
                    paths.append(list(path))
                return
            
            # Pruning: if current path length + min possible remaining > shortest_len, prune
            # But we don't have min possible remaining easily. 
            # However, since we are looking for shortest paths, we can just restrict depth.
            
            for nb in adj[cur]:
                if nb in visited:
                    continue
                if nb in banned and nb != v:
                    continue
                # Optimization: if we are at depth d, and dist[nb] is not defined or dist[nb] < dist[cur]+1?
                # Actually, BFS dist is from u. 
                # If we follow a path, the length is increasing.
                # We just need to ensure we don't exceed shortest_len.
                if len(path) - 1 >= shortest_len:
                    continue
                
                visited.add(nb)
                path.append(nb)
                dfs(nb, path, visited)
                path.pop()
                visited.remove(nb)
                
        dfs(u, [u], {u})
        return paths
