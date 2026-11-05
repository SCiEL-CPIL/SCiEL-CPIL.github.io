<<<<<<< SEARCH
def _find_longest_cycle(adj, start, n, max_search_nodes=5000):
    """Find a long cycle containing `start` using DFS with backtracking."""
    best_cycle = None
    best_len = 2  # Need at least 3 vertices for a cycle
    
    def dfs(u, visited, path):
        nonlocal best_cycle, best_len
        if len(path) > best_len:
            # Check if we can close the cycle back to start
            if start in adj[u]:
                candidate = path + [start]
                if len(candidate) > best_len:
                    best_len = len(candidate)
                    best_cycle = candidate[:]
        
        if len(path) >= n:  # Can't have longer than n vertices
            return
        
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                path.append(v)
                dfs(v, visited, path)
                path.pop()
                visited.remove(v)
            elif v == start and len(path) >= 2:
                # Found a cycle closing back to start
                candidate = path + [start]
                if len(candidate) > best_len:
                    best_len = len(candidate)
                    best_cycle = candidate[:]
    
    visited = {start}
    path = [start]
    dfs(start, visited, path)
    
    return best_cycle

def _find_cycle_greedy(adj, n, deadline):
    """Find a cycle by starting from high-degree vertices, preferring longer cycles."""
    best_cycle = None
    best_len = 0
    
    # Try vertices in order of decreasing degree
    vertices = sorted(range(n), key=lambda v: -len(adj[v]))
    
    for start in vertices:
        if time.time() > deadline - 2:
            break
        if not adj[start] or len(adj[start]) < 2:
            continue
        
        # First try to find a long cycle via DFS
        cycle = _find_longest_cycle(adj, start, n)
        if cycle is not None and len(cycle) > best_len:
            best_len = len(cycle)
            best_cycle = cycle
            # If we found a very long cycle, stop early
            if best_len >= n * 0.6:
                break
        elif best_cycle is None:
            # Fallback: find any cycle via BFS
            for nb in adj[start]:
                visited = {start: None, nb: nb}
                queue = deque([nb])
                found = False
                while queue and not found:
                    u = queue.popleft()
                    for v in adj[u]:
                        if v == start:
                            path = [u]
                            cur = u
                            while cur != nb:
                                cur = visited[cur]
                                path.append(cur)
                            path.reverse()
                            cycle = [start] + path
                            if len(cycle) > best_len:
                                best_len = len(cycle)
                                best_cycle = cycle
                            found = True
                            break
                        if v not in visited:
                            visited[v] = u
                            queue.append(v)
                if found:
                    break
    
    return best_cycle
=======
def _find_any_cycle(adj, n):
    """
    Find any cycle in the graph using BFS/DFS.
    Returns a list of vertices representing the cycle, or None if no cycle exists.
    """
    visited = {}
    # 0: unvisited, 1: visiting (in current stack), 2: visited (finished)
    state = {v: 0 for v in range(n)}
    parent = {v: None for v in range(n)}
    
    def dfs(u):
        state[u] = 1
        for v in adj[u]:
            if state[v] == 0:
                parent[v] = u
                cycle = dfs(v)
                if cycle:
                    return cycle
            elif state[v] == 1:
                # Found a back edge to a node in the current path
                # Reconstruct the cycle from v to u
                cycle = [u]
                cur = u
                while cur != v:
                    cur = parent[cur]
                    cycle.append(cur)
                cycle.reverse()
                return cycle
        state[u] = 2
        return None

    for v in range(n):
        if state[v] == 0:
            cycle = dfs(v)
            if cycle:
                return cycle
    return None
>>>>>>> REPLACE

