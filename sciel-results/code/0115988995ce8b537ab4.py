import time
import random
import sys
from collections import deque

def _has_girth_at_least(adj, n, target_girth, sample_size=100):
    """Check if girth >= target_girth using BFS from a random sample of vertices."""
    max_depth = target_girth // 2
    if n <= sample_size:
        starts = list(range(n))
    else:
        starts = random.sample(range(n), sample_size)
    for start in starts:
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        while qi < len(queue):
            v = queue[qi]
            qi += 1
            if dist[v] >= max_depth:
                continue
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    parent[u] = v
                    if dist[u] < max_depth:
                        queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    cycle_len = dist[v] + dist[u] + 1
                    if cycle_len < target_girth:
                        return False
    return True

def _construct_luw(k, g, d):
    """Construct LUW graph with parameters k, g, d. Returns (n, edges) or None if invalid."""
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
    
    rule = {}
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Verify girth
    if _has_girth_at_least(adj, n, g):
        return n, edges
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    if k == 3 and g == 16:
        # Try LUW with increasing d until we find a valid graph
        for d in range(4, 13):
            if time.time() > deadline - 15:
                break
            result = _construct_luw(k, g, d)
            if result is not None:
                return result
        
        # Fallback
        result = _construct_luw(k, g, 11)
        if result is not None:
            return result
            
    # Generic fallback
    d = g // 2 + 2
    result = _construct_luw(k, g, d)
    if result is not None:
        return result
    
    raise Exception("Failed to construct graph")
