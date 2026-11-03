    def has_cycle_shorter_than(adj, g):
        n = len(adj)
        # Check cycles of length 3 to g-1
        # For each start node, do DFS with path tracking up to depth g-1
        # This is O(N * 2^(g-1)) which is acceptable for g=16, N~1000.
        
        for start in range(n):
            # state: (current_node, prev_node, depth)
            # We want to find if we can return to start or any node in path.
            # Actually, just checking if we can return to start is enough for odd cycles?
            # No, even cycles don't necessarily return to start in the walk definition if we consider directed walks?
            # A non-backtracking walk of length L from v to v is a cycle of length L.
            # A non-backtracking walk of length L from v to w (w != v) is not a cycle by itself,
            # but if w is in the path, it is.
            
            # Efficient way:
            # For each v, count non-backtracking walks of length L.
            # If count > 0 for L < g, it might be a cycle or a path.
            # We need to distinguish.
            
            # Let's use the "unfolding" check.
            # A graph has girth >= g iff the non-backtracking matrix B has no eigenvalues... no.
            
            # Let's just do the DFS on the state space (u, p).
            # Start: (start, -1).
            # If we reach (start, prev) with depth >= 3, we have a cycle.
            # If we reach (u, p) where u is in the current path, we have a cycle.
            
            path = []
            visited_states = set() # (u, p)
            
            def dfs(u, p, depth):
                if depth >= g: return False
                if u == start and depth >= 3:
                    return True
                # Check if u is in path
                if u in path:
                    return True
                
                path.append(u)
                visited_states.add((u, p))
                
                for v in adj[u]:
                    if v == p: continue
                    if (v, u) in visited_states: # Pruning? No, different paths can hit same state.
                         # Actually, if we hit a visited state, it means we formed a cycle?
                         # Not necessarily.
                         pass
                    if dfs(v, u, depth + 1):
                        return True
                
                path.pop()
                visited_states.remove((u, p))
                return False
            
            if dfs(start, -1, 1):
                return True
        return False
    