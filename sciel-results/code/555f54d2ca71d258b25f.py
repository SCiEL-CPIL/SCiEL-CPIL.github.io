    def _find_random_cycle(adj, n, rng):
        # Try to find a cycle using randomized DFS
        # We want to maximize length, so we prioritize deep exploration.
        
        def dfs(u, path, visited):
            # If we can close the loop and path length >= 3
            if len(path) >= 3 and u in adj and start in adj[u]:
                # Found a cycle: path + [start]
                # But wait, u is the current node. If start is in adj[u], we can close.
                # Return the cycle
                return path[:] # The caller will add start? No, path is [start, ..., u]. Cycle is path + [start]
            
            # Try neighbors
            neighbors = list(adj[u])
            rng.shuffle(neighbors)
            
            for v in neighbors:
                if v == start and len(path) < 3:
                    continue # Need at least 3 nodes for a cycle (start, a, b, start)
                if v not in visited:
                    visited.add(v)
                    path.append(v)
                    res = dfs(v, path, visited)
                    if res:
                        return res
                    path.pop()
                    visited.remove(v)
            
            return None

        # Pick a random start
        if not adj: return None
        start = rng.choice([i for i in range(n) if adj[i]])
        
        # To find a *long* cycle, simple DFS might get stuck in a small loop.
        # A better heuristic for "longest cycle" approximation:
        # 1. Find a longest path in the graph (approx).
        # 2. If endpoints are connected, it's a cycle.
        
        # Let's use a heuristic: Randomized DFS that prefers to go deep.
        # We can also try multiple starts.
        
        best_cycle = None
        best_len = 0
        
        for _ in range(20): # Try 20 random starts/paths
            start = rng.choice([i for i in range(n) if adj[i]])
            visited = {start}
            path = [start]
            
            # Iterative DFS with backtracking to find a cycle
            # This simple recursive DFS above is okay for n=30.
            # But it returns the *first* cycle found.
            # To get a *longer* cycle, we can modify the DFS to not stop at the first cycle,
            # but keep exploring? No, that's complex.
            
            # Alternative: Find a long path first.
            # Let's just use the recursive DFS but with a twist:
            # We can bias the neighbor selection.
            
            res = dfs(start, [start], {start})
            if res:
                # res is [start, ..., u]. The cycle is res + [start]
                # Wait, my dfs returns `path` which is [start, ..., u].
                # The cycle vertices are [start, ..., u].
                # Edges: (start, v1), ..., (u, start).
                # So the cycle piece is `res`.
                # Length is len(res).
                if len(res) > best_len:
                    best_len = len(res)
                    best_cycle = res[:]
            
        return best_cycle
    