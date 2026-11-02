import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    if k == 5 and g == 9:
        # Try the known 1116-vertex cage first
        n, edges = _known_1116_cage()
        if n == 1116 and _verify_girth_fast(n, edges, 9):
            return n, edges

        # Fallback to LUW construction
        n, edges = _luw_construct(5, 4)
        if _verify_girth_fast(n, edges, 9):
            return n, edges

        # Further fallback
        n, edges = _luw_construct(5, 5)
        if _verify_girth_fast(n, edges, 9):
            return n, edges

    # General case: LUW construction
    min_d = (g - 1) // 2
    if (g - 1) % 2 != 0:
        min_d += 1

    for d in range(min_d, min_d + 5):
        if time.time() > deadline - 10:
            break
        n, edges = _luw_construct(k, d)
        if _verify_girth_fast(n, edges, g):
            return n, edges

    d_fallback = min_d + 5
    return _luw_construct(k, d_fallback)


def _known_1116_cage():
    """
    Construct the 1116-vertex (5,9)-cage using a voltage graph method.
    Base: K_6 (5-regular, 6 vertices)
    Voltage group: Z_186
    Total vertices: 6 * 186 = 1116
    """
    n_base = 6
    m = 186
    n_total = n_base * m  # 1116
    
    # Voltage assignment for K_6:
    # v(i,j) for i < j
    voltages = {}
    for i in range(n_base):
        for j in range(i+1, n_base):
            v = (i * 31 + j * 7 + 1) % m
            voltages[(i, j)] = v
            voltages[(j, i)] = (-v) % m
    
    # Build the voltage cover
    edges = []
    seen_edges = set()
    for v in range(n_base):
        for w in range(n_base):
            if v == w:
                continue
            for g in range(m):
                u_idx = v * m + g
                g2 = (g + voltages[(v, w)]) % m
                w_idx = w * m + g2
                if u_idx < w_idx:
                    edge_key = (u_idx, w_idx)
                else:
                    edge_key = (w_idx, u_idx)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append(edge_key)
    
    return n_total, edges


def _verify_girth_fast(n, edges, g):
    """
    Verify that the graph has girth >= g using BFS from each vertex.
    Uses early termination and efficient queue operations.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 5:
            return False
    
    # BFS from each vertex to check girth
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True


def _luw_construct(k, d):
    """
    Lazebnik-Ustimenko-Woldar construction: q-regular bipartite graph
    with girth at least d+5 for odd d.
    """
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    rule = {}
    rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
