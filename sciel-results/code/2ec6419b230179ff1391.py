"""
Construct a k-regular graph with girth at least g.
Strategy:
1. For k=3, g=16: Try voltage lifts of the Petersen graph with many random trials
   to find a lift with small n (m * 10) and girth >= 16.
2. Fallback to LUW construction.
"""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin out of 150s limit

    def check_deadline():
        return time.time() > deadline

    if k == 3 and g == 16:
        # Try voltage lifts of Petersen graph
        for m in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]:
            if check_deadline():
                break
            n_total = m * 10
            # Try many random lifts
            max_tries = 500 if m <= 100 else 200
            for _ in range(max_tries):
                if check_deadline():
                    break
                try:
                    n, edges = _build_lift(m, g)
                    if n is not None and _verify_girth_fast(n, edges, g):
                        return n, edges
                except Exception:
                    pass
        # Fallback to LUW
        return _luw_construct(k, g)
    
    # For other k, g: fallback to LUW
    return _luw_construct(k, g)


def _build_lift(m, g):
    """
    Build a random voltage lift of the Petersen graph over Z_m.
    Returns (n, edges) or (None, None) if not valid.
    """
    # Petersen graph edges
    petersen_edges = [
        (0, 1), (0, 4), (0, 5),
        (1, 2), (1, 9),
        (2, 3), (2, 8),
        (3, 4), (3, 7),
        (4, 9),
        (5, 6), (5, 7),
        (6, 8), (6, 9),
        (7, 8)
    ]
    
    # Assign random voltages
    voltages = {}
    for u, v in petersen_edges:
        voltages[(u, v)] = random.randint(0, m - 1)
        voltages[(v, u)] = (-voltages[(u, v)]) % m
    
    n = m * 10
    edges = []
    
    for u, v in petersen_edges:
        for i in range(m):
            j = (i + voltages[(u, v)]) % m
            u_id = i * 10 + u
            v_id = j * 10 + v
            edges.append((u_id, v_id))
    
    return n, edges


def _verify_girth_fast(n, edges, g):
    """
    Fast girth verification using BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return False
    
    # BFS from each vertex to find shortest cycle
    for src in range(n):
        if _bfs_short_cycle(src, adj, n, g):
            return False
    
    return True


def _bfs_short_cycle(src, adj, n, g):
    """
    BFS from src to find if there's a cycle shorter than g.
    Returns True if a short cycle is found.
    """
    dist = [-1] * n
    parent = [-1] * n
    dist[src] = 0
    queue = deque([src])
    
    max_depth = g // 2
    
    while queue:
        u = queue.popleft()
        depth = dist[u]
        
        if depth >= max_depth:
            # Still check edges from u for back edges to shallow nodes
            for v in adj[u]:
                if parent[v] == u or parent[u] == v:
                    continue
                if dist[v] != -1:
                    cycle_len = depth + 1 + dist[v]
                    if cycle_len < g:
                        return True
            continue
        
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = depth + 1
                parent[v] = u
                queue.append(v)
            else:
                # Back edge - check for cycle
                if parent[v] == u or parent[u] == v:
                    continue
                cycle_len = depth + 1 + dist[v]
                if cycle_len < g:
                    return True
    
    return False


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # make it odd
    
    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with specified d."""
    q = k
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

# EVOLVE-BLOCK-END
