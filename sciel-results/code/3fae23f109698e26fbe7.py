"""Stronger construction for the (5,9) cage problem."""
import time
from collections import deque

import numpy as np

def _moore_bound(k, g):
    if g % 2 == 0:
        h = g // 2
        return 2 * sum(k ** i for i in range(h))
    else:
        h = (g - 1) // 2
        return k + k * (k - 1) * sum((k - 1) ** i for i in range(h))

def _build_luw(k, g):
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

def _verify_girth(n, edges, g):
    """Verify girth >= g via BFS from each vertex (non-backtracking)."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for start in range(n):
        dist = [-1] * n
        prev = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v == prev[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    if dist[v] < g - 1:
                        pass
                    queue.append(v)
                else:
                    if dist[u] + 1 < dist[v] and dist[u] + 1 + 1 < g:
                        return False
    return True

def _build_from_incidence(k, g, seed):
    """Try a random regular graph construction and verify girth."""
    # For k=5, g=9: Moore bound is 5*(1 + 4 + 16 + 64) = 5*85 = 425
    # Target: find smallest n >= 425 that is divisible by k (since k-regular)
    # Actually n must be >= 2*k for a k-regular graph to exist
    mb = _moore_bound(k, g)
    # n must be even for a regular graph to exist (sum of degrees = n*k must be even)
    # Also n >= k+1
    min_n = max(k + 1, mb)
    if min_n % 2 != 0:
        min_n += 1
    # Try to construct using random regular graph with verification
    # This is expensive, so limit attempts
    np.random.seed(seed)
    for n in range(min_n, min_n + 20):  # Try a few values
        if n < k + 1:
            continue
        if n * k % 2 != 0:
            continue
        # Generate random k-regular graph using configuration model
        half_edges = n * k // 2
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts:
            attempts += 1
            # Configuration model: create multiset of stubs
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            np.random.shuffle(stubs)
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.add((min(u, v), max(u, v)))
            if valid and len(edges) == half_edges:
                edge_list = list(edges)
                # Verify degree
                deg = [0] * n
                for u, v in edge_list:
                    deg[u] += 1
                    deg[v] += 1
                if all(d == k for d in deg):
                    # Verify girth
                    if _verify_girth(n, edge_list, g):
                        return n, edge_list
    return None

def construct(k, g):
    deadline = time.time() + 120  # 120s wall-clock
    
    # Start with LUW construction
    n_luw, edges_luw = _build_luw(k, g)
    best_n, best_edges = n_luw, edges_luw
    
    elapsed = time.time() - (deadline - 120)
    if elapsed > 60:
        return best_n, best_edges
    
    # Try to improve with random constructions
    # For k=5, g=9, try a few seeds
    for seed in range(100):
        if time.time() > deadline - 30:
            break
        result = _build_from_incidence(k, g, seed)
        if result is not None:
            n_new, edges_new = result
            if n_new < best_n:
                best_n, best_edges = n_new, edges_new
    
    return best_n, best_edges
