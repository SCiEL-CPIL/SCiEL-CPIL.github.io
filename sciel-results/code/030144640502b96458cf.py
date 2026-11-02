import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Moore Bound
    # n >= 2 * (k^(g/2 - 1) - 1) / (k - 1) for even g
    if g % 2 == 0:
        n_min = 2 * (k ** (g // 2 - 1) - 1) // (k - 1)
    else:
        # For odd g, Moore bound is different, but for k=4, g=10 is even
        n_min = 2 * (k ** ((g - 1) // 2) - 1) // (k - 1)
        
    # Ensure n_min is at least k+1
    n_min = max(n_min, k + 1)
    
    # If k=4, g=10, n_min = 2 * (4^4 - 1) / 3 = 2 * 255 / 3 = 170
    
    # Try to find a graph with n from n_min to 400
    # We prioritize smaller n
    
    for n in range(n_min, 401):
        if time.time() > deadline:
            break
        # Try to find a valid graph for this n
        # Number of attempts depends on time
        attempts = 0
        while time.time() < deadline and attempts < 500:
            attempts += 1
            graph = _generate_random_regular_graph(n, k)
            if graph is None:
                continue
            if _is_connected(graph, n):
                girth = _calculate_girth(graph, n, g)
                if girth >= g:
                    # Convert adjacency list to edge list
                    edges = []
                    for u in range(n):
                        for v in graph[u]:
                            if u < v:
                                edges.append((u, v))
                    return n, edges
        # If we didn't find one for this n, try next n
        
    # Fallback to LUW construction
    return _luw_construct(k, g)

def _generate_random_regular_graph(n, k):
    # Configuration model
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    
    for _ in range(100): # Try up to 100 times to get a simple graph
        random.shuffle(stubs)
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if valid:
            # Check degree
            for i in range(n):
                if len(adj[i]) != k:
                    valid = False
                    break
            if valid:
                return [list(s) for s in adj]
    return None

def _is_connected(adj, n):
    visited = [False] * n
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        u = queue.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    return count == n

def _calculate_girth(adj, n, max_girth):
    # BFS from each vertex
    # If we find a cycle of length L < max_girth, return L
    # We only care if girth >= max_girth
    # So we can stop BFS at depth max_girth // 2
    half_g = max_girth // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= half_g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    # Length = dist[u] + dist[v] + 1
                    # But we need to be careful with BFS tree
                    # This simple check might find non-simple cycles or double count
                    # However, for girth check, if we find ANY cycle < max_girth, we fail.
                    # A safer way: check if dist[u] + dist[v] + 1 < max_girth
                    # But this might not be the shortest cycle.
                    # However, if the shortest cycle is < max_girth, it will be found.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < max_girth:
                        return cycle_len
    return max_girth # Or infinity

def _luw_construct(k, g):
    # Original LUW code
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
