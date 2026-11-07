"""Improved program: greedy extraction of edge-disjoint Hamiltonian cycles."""

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(edges):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj

def _find_ham_cycle(adj, start, n, time_limit):
    """Find a Hamiltonian cycle starting at 'start' using Warnsdorff heuristic."""
    import time
    deadline = time.time() + time_limit
    visited = [False] * n
    path = [start]
    visited[start] = True

    def dfs(path, visited):
        import time
        if time.time() > deadline:
            return None
        if len(path) == n:
            # Check if last connects back to start
            if path[-1] in adj.get(start, set()):
                return list(path)
            return None
        cur = path[-1]
        neighbors = list(adj.get(cur, set()))
        # Warnsdorff: sort by number of unvisited neighbors (fewest first)
        neighbors.sort(key=lambda v: sum(1 for w in adj.get(v, set()) if not visited[w]))
        for nxt in neighbors:
            if not visited[nxt]:
                visited[nxt] = True
                path.append(nxt)
                result = dfs(path, visited)
                if result is not None:
                    return result
                path.pop()
                visited[nxt] = False
        return None

    return dfs(path, visited)

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s kill

    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = _build_adj(edges)
    
    # Track used edges (as frozenset pairs)
    used_edges = set()
    cycles = []
    
    # Remaining adjacency (dynamic)
    rem_adj = {v: set() for v in range(n)}
    for v in range(n):
        rem_adj[v] = set(adj.get(v, set()))
    
    # Try multiple starting vertices and orderings
    # First pass: try each vertex as start
    for attempt in range(50):
        if time.time() > deadline:
            break
        if len(cycles) >= 10:
            break
        
        # Pick a start vertex - prefer vertices with high remaining degree
        candidates = list(range(n))
        candidates.sort(key=lambda v: len(rem_adj[v]), reverse=True)
        
        found = False
        for start in candidates:
            if len(rem_adj[start]) < 2:
                continue
            
            # Try to find Hamiltonian cycle using only remaining edges
            time_per_try = max(0.1, (deadline - time.time()) / 10.0)
            cycle = _find_ham_cycle(rem_adj, start, n, time_per_try)
            
            if cycle is not None:
                # Verify cycle uses only remaining edges
                valid = True
                for i in range(len(cycle)):
                    a = cycle[i]
                    b = cycle[(i + 1) % len(cycle)]
                    if b not in rem_adj[a]:
                        valid = False
                        break
                if valid:
                    # Remove these edges from remaining graph
                    for i in range(len(cycle)):
                        a = cycle[i]
                        b = cycle[(i + 1) % len(cycle)]
                        rem_adj[a].discard(b)
                        rem_adj[b].discard(a)
                        used_edges.add(frozenset((a, b)))
                    cycles.append(list(cycle))
                    found = True
                    break
        
        if not found:
            # No more cycles can be found
            break
    
    # Ensure we have at least one cycle (fallback)
    if not cycles:
        # Try a simpler approach: just return a single cycle if we can find one
        adj_full = _build_adj(edges)
        cycle = _find_ham_cycle(adj_full, 0, n, 5.0)
        if cycle is not None:
            cycles.append(list(cycle))
    
    # Convert to ensure plain Python ints and lists
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    
    return result if result else [list(range(n))]
