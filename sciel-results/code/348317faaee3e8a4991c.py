import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    if k == 4 and g == 10:
        # Voltage Graph Construction on K5 with Z8 x Z8
        base_edges = [(i, j) for i in range(5) for j in range(i+1, 5)]
        m = 8
        l = 8
        n_lift = 5 * m * l
        
        def check_girth_fast(edges, n, girth_target):
            """Check if graph has no cycle shorter than girth_target.
            Returns True if girth >= girth_target, False otherwise."""
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # For each vertex, BFS to find shortest cycle through it
            # A cycle through start: find two neighbors a,b of start such that
            # shortest path from a to b (not through start) + 2
            # More generally: BFS from start, track dist and parent
            # When we find a back edge (u,v) where v != parent[u] and u != parent[v],
            # cycle length = dist[u] + dist[v] + 1
            
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = deque([start])
                
                while queue:
                    u = queue.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u] and u != parent[v]:
                            # Potential cycle
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < girth_target:
                                return False
                # Early exit not needed per-vertex, but we check all
            return True
        
        def build_lift(voltages):
            """voltages: list of (a,b) for each of the 10 base edges"""
            edges = []
            ml = m * l
            for idx, (u, v) in enumerate(base_edges):
                a, b = voltages[idx]
                for i in range(m):
                    for j in range(l):
                        g2_i = (i + a) % m
                        g2_j = (j + b) % l
                        u_node = u * ml + i * l + j
                        v_node = v * ml + g2_i * l + g2_j
                        edges.append((u_node, v_node))
            return edges
        
        best_edges = None
        best_girth = 0
        random.seed(42)
        
        # Random search with early termination
        attempts = 0
        while time.time() < deadline - 5:
            voltages = [(random.randint(0, m-1), random.randint(0, l-1)) for _ in range(10)]
            edges = build_lift(voltages)
            
            if check_girth_fast(edges, n_lift, 10):
                return n_lift, edges
            
            attempts += 1
            
            # Periodic check
            if attempts % 50 == 0 and time.time() > deadline - 10:
                break
        
        # Fallback: LUW Construction
        return _luw_construct(k, g)
    
    return _luw_construct(k, g)


def _luw_construct(k, g):
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
        
    # Fallback: LUW Construction from seed
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
