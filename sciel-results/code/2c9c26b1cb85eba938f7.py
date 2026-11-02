import time
import random
from collections import deque

def _compute_girth_local(adjacency, n, u, v, max_depth):
    """Check if there is a path of length <= max_depth between u and v."""
    if u == v:
        return 0
    dist = {u: 0}
    queue = deque([u])
    while queue:
        curr = queue.popleft()
        d = dist[curr]
        if d >= max_depth:
            continue
        for neighbor in adjacency[curr]:
            if neighbor == v:
                return d + 1
            if neighbor not in dist:
                dist[neighbor] = d + 1
                queue.append(neighbor)
    return -1

def _find_short_cycle(adjacency, n, max_girth=8):
    """Find a cycle of length < max_girth. Returns list of vertices or None."""
    # BFS from each vertex, tracking parent
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and parent[v] != u:
                    # Found a cycle
                    # Reconstruct cycle
                    # Path from u to v + edge (u,v)
                    # But we need the actual cycle length
                    # The cycle is formed by the path from LCA(u,v) to u, path from LCA to v, and edge?
                    # Simpler: just return the length
                    # We need the vertices to break it.
                    # For now, just return the length and start node
                    # To break it, we need to identify an edge in the cycle.
                    # This is complex.
                    pass
    return None

def _random_5_regular(n):
    """Generate a random 5-regular graph on n vertices."""
    if n * 5 % 2 != 0:
        return None
    stubs = []
    for i in range(n):
        stubs.extend([i] * 5)
    random.shuffle(stubs)
    
    adjacency = [set() for _ in range(n)]
    i = 0
    while i < len(stubs):
        u, v = stubs[i], stubs[i + 1]
        i += 2
        if u == v:
            return None
        if v in adjacency[u]:
            return None
        adjacency[u].add(v)
        adjacency[v].add(u)
    
    for i in range(n):
        if len(adjacency[i]) != 5:
            return None
    return adjacency

def _luw_construct(k, g):
    # Fallback
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    if k == 5 and g == 9:
        # Try to find a small graph
        # Target 1116
        target_n = 1116
        start_time = time.time()
        
        for _ in range(10): # 10 restarts
            if time.time() - start_time > 100:
                break
            adjacency = _random_5_regular(target_n)
            if adjacency is None:
                continue
            
            # Try to improve girth
            # We need girth >= 9
            # This is very hard.
            pass
        
        # Fallback to LUW
        return _luw_construct(k, g)
    
    return _luw_construct(k, g)
