import time
import sys
import collections

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    if k != 3:
        return 0, []

    q = k  # q = 3

    # Determine d such that girth >= g
    def girth_of_d(d):
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6

    d = 3
    while girth_of_d(d) < g:
        d += 1

    # GF(3) operations
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return (x * y) % q

    # LUW rule
    rule = {}
    for c in range(2, d + 1):
        if c == 2:
            rule[c] = (1, 1)
        elif c == 3:
            rule[c] = (2, 1)
        elif c == 4:
            rule[c] = (1, 2)
        else:
            rule[c] = (1, c - 2)

    # Generate vertices
    index = {}
    order = []
    edges = []
    
    start_vertex = (0, (None,) + (0,) * d)
    index[start_vertex] = 0
    order.append(start_vertex)
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            neighbor = (1 - side, tuple(y))
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            
            if side == 0:
                edges.append((index[vertex], index[neighbor]))
    
    n = len(order)
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    edge_set = set()
    for (u, v) in edges:
        if u == v:
            continue
        key = (min(u,v), max(u,v))
        if key in edge_set:
            continue
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    
    # Check k-regular
    valid = True
    for i in range(n):
        if len(adj[i]) != k:
            valid = False
            break
    
    if not valid:
        return 0, []
    
    # Fast girth check using non-backtracking BFS
    def has_short_cycle(adj, n, g):
        half_g = g // 2
        for start in range(n):
            visited = {}
            queue = collections.deque()
            queue.append((start, -1, 0))
            visited[(start, -1)] = 0
            while queue:
                node, prev, dist = queue.popleft()
                if dist >= half_g:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    state = (nb, node)
                    if state not in visited:
                        visited[state] = dist + 1
                        queue.append((nb, node, dist + 1))
                    elif dist + 1 > visited[state]:
                        cycle_len = dist + 1 + visited[state]
                        if cycle_len < g:
                            return True
        return False
    
    if has_short_cycle(adj, n, g):
        return 0, []
    
    return n, list(edge_set)
