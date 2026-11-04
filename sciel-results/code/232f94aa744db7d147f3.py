import time
import random
from collections import deque

_DEADLINE = 115.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
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

def _generate_random_4regular(n):
    if n % 2 != 0:
        return None
    half_edges = []
    for i in range(n):
        for _ in range(4):
            half_edges.append(i)
    random.shuffle(half_edges)
    edges = set()
    for i in range(0, len(half_edges), 2):
        u, v = half_edges[i], half_edges[i+1]
        if u == v:
            return None
        key = (min(u, v), max(u, v))
        if key in edges:
            return None
        edges.add(key)
    if len(edges) != n * 2:
        return None
    return list(edges)

def _get_adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _find_short_cycle(n, adj, max_girth):
    for start in range(n):
        dist = {start: (None, 0)}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            prev, d = dist[u]
            for v in adj[u]:
                if v == prev:
                    continue
                if v in dist:
                    cycle_len = dist[u][1] + dist[v][1] + 1
                    if cycle_len < max_girth:
                        return cycle_len
                else:
                    dist[v] = (u, dist[u][1] + 1)
                    queue.append(v)
    return max_girth

def _try_repair(n, edges, g, time_limit):
    adj = _get_adjacency(n, edges)
    start_time = time.time()
    while time.time() - start_time < time_limit:
        girth = _find_short_cycle(n, adj, g)
        if girth >= g:
            return edges
        # Find a cycle and break it
        # Simplified: Random 2-switch
        # Pick random edge
        e1 = random.choice(edges)
        u1, v1 = e1
        # Pick another edge
        e2 = random.choice(edges)
        u2, v2 = e2
        # Ensure they are not the same or adjacent in a way that breaks simplicity
        if e1 == e2:
            continue
        # Try 2-switch: (u1, v1), (u2, v2) -> (u1, u2), (v1, v2) or (u1, v2), (v1, u2)
        # Check for self-loops and multi-edges
        new_e1 = (min(u1, u2), max(u1, u2))
        new_e2 = (min(v1, v2), max(v1, v2))
        if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
            continue
        if new_e1 in edges or new_e2 in edges:
            continue
        if new_e1 == new_e2:
            continue
        
        # Apply
        edges.remove(e1)
        edges.remove(e2)
        edges.add(new_e1)
        edges.add(new_e2)
        # Update adj
        adj[u1].remove(v1); adj[v1].remove(u1)
        adj[u2].remove(v2); adj[v2].remove(u2)
        adj[u1].append(u2); adj[u2].append(u1)
        adj[v1].append(v2); adj[v2].append(v1)
        
    return None

def construct(k, g):
    global _START
    _START = time.time()
    
    n_luw, edges_luw = _luw(k, g)
    
    if k == 4 and g == 10:
        # Try to find a smaller graph
        if _remaining() > 30:
            # Try N=320
            for target_n in [320]:
                for _ in range(10):
                    if _remaining() < 10:
                        break
                    edges = _generate_random_4regular(target_n)
                    if edges is None:
                        continue
                    # Try to repair
                    repaired = _try_repair(target_n, edges, g, 5.0)
                    if repaired is not None:
                        return target_n, repaired
                        
    return n_luw, edges_luw
