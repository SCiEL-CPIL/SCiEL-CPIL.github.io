<<<<<<< SEARCH
def _find_short_cycle(adj, g):
    n = len(adj)
    # Start from a random vertex to diversify
    start = random.randint(0, n - 1)
    visited = {start: (None, 0)}
    q = deque([(start, None, 0)])
    while q:
        u, prev, d = q.popleft()
        if d >= g - 1:
            continue
        for v in adj[u]:
            if v == prev:
                continue
            if v == start and d + 1 < g:
                # Found cycle
                path = [v, u]
                cur = u
                while cur != start:
                    cur = visited[cur][0]
                    path.append(cur)
                path.reverse()
                return path
            if v not in visited:
                visited[v] = (u, d + 1)
                q.append((v, u, d + 1))
    return None
=======
def _find_short_cycle(adj, g):
    n = len(adj)
    # Start from a random vertex to diversify
    start = random.randint(0, n - 1)
    # Non-backtracking BFS: visited[(vertex, prev)] = (prev_of_prev, depth)
    visited = {}
    q = deque()
    for neighbor in adj[start]:
        visited[(neighbor, start)] = (None, 1)
        q.append((neighbor, start, 1))
    while q:
        u, prev, d = q.popleft()
        if d + 1 >= g:
            continue
        for v in adj[u]:
            if v == prev:
                continue
            if v == start:
                # Found a cycle of length d+1
                path = [v, u]
                cur = u
                while cur != start:
                    prev_of_cur = visited[(cur, prev)][0]
                    path.append(prev_of_cur)
                    cur = prev_of_cur
                path.reverse()
                return path
            else:
                state = (v, u)
                if state not in visited:
                    visited[state] = (prev, d + 1)
                    q.append((v, u, d + 1))
    return None
>>>>>>> REPLACE

