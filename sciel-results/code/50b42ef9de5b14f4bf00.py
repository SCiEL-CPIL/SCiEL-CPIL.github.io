import random
import time

def construct(k, g):
    # k=4, g=10
    if k != 4 or g != 10:
        # Fallback to LUW if parameters are different
        return _luw_construct(k, g)
    
    deadline = time.time() + 110  # 110s limit
    
    # Try to find a random 4-regular graph with girth >= 10
    # Start with n=320, then 512, 1024, etc.
    # But 320 is the target.
    
    for n in [320, 512, 1024, 2048]:
        if time.time() > deadline:
            break
        # Try several random graphs
        for attempt in range(1000):
            if time.time() > deadline:
                break
            edges = _generate_random_4_regular(n)
            if edges is None:
                continue
            if _has_girth_at_least(edges, n, 10):
                return n, edges
    
    # If no random graph found, return LUW
    return _luw_construct(4, 10)

def _generate_random_4_regular(n):
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(4):
            stubs.append(i)
    random.shuffle(stubs)
    edges = []
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None # Self-loop
        edges.append((u, v))
    # Check for multi-edges
    edge_set = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return None
        edge_set.add(key)
    return edges

def _has_girth_at_least(edges, n, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # BFS from each vertex to depth g//2 - 1 ?
    # Actually, to find cycles of length < g, we need to check paths of length < g.
    # A cycle of length L is a path of length L from u to u.
    # We can do BFS from each vertex and check if we revisit a vertex at distance < g.
    # But this is O(n * 4^(g/2)).
    # For g=10, depth 5. 4^5 = 1024. n=320. 320*1024 = 327k ops.
    # This is fast.
    
    for start in range(n):
        # BFS
        dist = [-1] * n
        dist[start] = 0
        queue = [(start, -1)] # (node, parent)
        while queue:
            node, parent = queue.pop(0)
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    if dist[neighbor] >= g:
                        break
                    queue.append((neighbor, node))
                else:
                    # Found a cycle
                    # The length of the cycle is dist[node] + dist[neighbor] + 1
                    # But we need to be careful.
                    # If we found a back-edge, the cycle length is dist[node] + dist[neighbor] + 1.
                    # However, this might not be the shortest cycle.
                    # But if ANY cycle is < g, we return False.
                    cycle_len = dist[node] + dist[neighbor] + 1
                    if cycle_len < g:
                        return False
    return True

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
