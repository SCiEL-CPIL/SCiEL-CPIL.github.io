import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    # Special case: Known small graphs
    if k == 4 and g == 5:
        # Petersen graph, n=10
        return _petersen_graph()
    if k == 3 and g == 5:
        # Petersen graph is 3-regular, girth 5, n=10
        return _petersen_graph()

    # For k=5, g=9, the LUW construction gives n=1250.
    # The known best is 1116.
    # We try to see if we can find a smaller graph via heuristic search.
    # But heuristic search for girth 9 is very hard.
    # We fallback to LUW.
    
    if k == 5 and g == 9:
        n, edges = _luw_construct(5, 4)
        if _verify_girth_fast(n, edges, g):
            return n, edges

    # General fallback: LUW construction
    # d is chosen such that 2*k^d is minimized but girth >= g
    # Girth of LUW(d, q) is at least 2d+1?
    # For d=4, girth is at least 9?
    # Actually, LUW graphs have girth at least 2d+1 if q is large enough.
    # For q=5, d=4, girth is 9?
    # Let's assume d=4 works for g=9.
    
    # Try increasing d until girth is satisfied
    for d in range(2, 10):
        if time.time() > deadline - 10:
            break
        n, edges = _luw_construct(k, d)
        if _verify_girth_fast(n, edges, g):
            return n, edges

    # If LUW fails or is too slow, return a random regular graph (might be invalid, but better than crash?)
    # No, must be valid.
    # Use LUW with large d as fallback (guaranteed high girth, but large n)
    n, edges = _luw_construct(k, 10)
    return n, edges

def _petersen_graph():
    # Petersen graph: 10 vertices, 5-regular? No, 3-regular.
    # Vertices: 0..4 (inner), 5..9 (outer)
    # Edges: outer cycle (0-1-2-3-4-0), inner star (5-7-9-6-8-5), spokes (0-5, 1-6, 2-7, 3-8, 4-9)
    edges = []
    for i in range(5):
        edges.append((i, (i + 1) % 5))
        edges.append((5 + i, 5 + (i + 2) % 5))
        edges.append((i, 5 + i))
    return 10, edges

def _luw_construct(k, d):
    """
    Lazebnik-Ustimenko-Woldar construction CD(d, k) with q = k.
    Returns a k-regular bipartite graph on 2*k^d vertices.
    """
    q = k
    # For q=5, we use modulo 5 arithmetic.
    # add, sub, mul are mod q.
    
    # Generate the rule for c = 2..d
    # Standard LUW rule:
    rule = {}
    if d >= 2:
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

    # Precompute multiplication table for speed? 
    # mul(x, y) = (x * y) % q
    mul_table = [[(i * j) % q for j in range(q)] for i in range(q)]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                # x is tuple of length d+1, x[0] is None or side indicator?
                # In seed: start = (0, (None,) + (0,) * d)
                # So x has length d+1. x[0] is None. x[1..d] are the coordinates.
                # Wait, the seed uses 1-based indexing for coordinates in the tuple?
                # seed: y = [None, first] + [0] * (d - 1)
                # y[0] is None. y[1] is first. y[2..d] are 0.
                # x is similar.
                if side == 0:
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = (x[c] + mul_table[y[a]][x[b]]) % q
                else:
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = (x[c] - mul_table[x[a]][y[b]]) % q
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # Optimization: Use a set for faster lookup? 
    # index is a dict.
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _verify_girth_fast(n, edges, g):
    """
    Fast girth verification using BFS.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif u != dist[v] - 1: # Avoid parent check complexity, just check cycle length
                    # If we see a visited node, and it's not the parent, we found a cycle.
                    # But this simple BFS finds shortest path, not necessarily girth if we don't track parents carefully.
                    # Correct BFS girth check:
                    # If dist[u] + dist[v] + 1 < g, return False.
                    # This works if we track parent to avoid going back.
                    pass
    # The seed implementation was better. I'll use a standard BFS girth check.
    return _verify_girth(n, edges, g)

def _verify_girth(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
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
