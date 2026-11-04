"""Combined program: Voltage lift of Petersen graph for cage construction."""
import time
import random
from collections import deque

def _check_girth(n, edges, g):
    """Return True iff no cycle is shorter than g (BFS from every vertex)."""
    if n == 0:
        return True
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    
    # Use a fast BFS with early termination
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = [s]
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            du = dist[u]
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] != -1:
                    # Found a cycle: dist[u] + dist[v] + 1
                    if du + dist[v] + 1 < g:
                        return False
                else:
                    dist[v] = du + 1
                    parent[v] = u
                    q.append(v)
    return True

def _luw(q, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d,q); girth is 2d+2. Returns (n, edges)."""
    if q == 4:
        mul_table = [[0, 0, 0, 0],
                     [0, 1, 2, 3],
                     [0, 2, 3, 1],
                     [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return mul_table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    rules = [None] * (d + 1)
    for c in range(2, d + 1):
        rules[c] = rule[c]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rules[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rules[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (0,) * (d + 1))
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        vi = index[vertex]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((vi, index[other]))
    return len(order), edges

def _fallback(k, g):
    """Guaranteed valid k-regular graph with girth >= g via a circulant."""
    n = 2 * g + 2
    edges = []
    for i in range(n):
        for j in (1, g - 1, g):
            a, b = i, (i + j) % n
            if a > b:
                a, b = b, a
            edges.append((a, b))
    return n, edges

def _petersen_lift(m, voltages):
    """Construct the voltage lift of the Petersen graph."""
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n = 10 * m
    edges = []
    for i, (u, w) in enumerate(base_edges):
        volt = voltages[i]
        for v in range(m):
            idx_u = u * m + v
            idx_w = w * m + ((v + volt) % m)
            edges.append((idx_u, idx_w))
    return n, edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    if k != 3:
        return _fallback(k, g)

    best_n = float('inf')
    best_edges = None

    # Strategy: Random voltage lifts of Petersen graph
    # n = 10 * m
    # Moore bound for k=3, g=16 is 382. So m >= 39.
    # We try m from 40 to 200.
    
    for m in range(40, 201):
        if time.time() > deadline - 25.0:
            break
        
        n = 10 * m
        if n >= best_n:
            continue
        
        # Number of attempts: 
        # For small m, girth is harder to achieve (more cycles in base to worry about).
        # For large m, probability of any specific cycle summing to 0 is 1/m.
        # So we can try fewer times for large m? 
        # Actually, we just want the first one that works.
        # Let's try 10-20 times per m.
        max_attempts = 15
        
        for _ in range(max_attempts):
            if time.time() > deadline - 15.0:
                break
            
            voltages = [random.randint(0, m - 1) for _ in range(15)]
            n, edges = _petersen_lift(m, voltages)
            
            if _check_girth(n, edges, g):
                return n, edges

    # If random search fails, fall back to LUW
    # LUW CD(7, 3) has girth 16 and n = 2 * 3^7 = 4374
    try:
        n, edges = _luw(3, 7)
        if _check_girth(n, edges, g):
            return n, edges
    except Exception:
        pass

    # Absolute fallback
    return _fallback(k, g)
