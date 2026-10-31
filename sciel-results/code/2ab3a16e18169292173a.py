<<<<<<< SEARCH
def _verify_girth(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise."""
    for start in range(n):
        # State: (vertex, prev_vertex), track min distance
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False  # found cycle shorter than g
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append((w, v))
    return True
=======
def _verify_girth_fast(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    Optimized for speed: stops early if a short cycle is found."""
    for start in range(n):
        # State: (vertex, prev_vertex)
        # We use a dictionary for distance. 
        # Since degree is 3, the number of states is bounded by 2*n*3 roughly.
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                # If we return to start, we have a cycle of length d+1
                # Note: d is the distance from start to v. 
                # If w == start, the cycle length is d+1.
                # We need cycle length >= g.
                if w == start:
                    if d + 1 < g:
                        return False
                else:
                    state = (w, v)
                    if state not in dist:
                        dist[state] = d + 1
                        dq.append(state)
    return True
>>>>>>> REPLACE

