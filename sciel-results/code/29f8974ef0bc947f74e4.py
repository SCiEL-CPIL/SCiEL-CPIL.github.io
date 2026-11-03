import time
import random
import sys
from collections import deque
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    if k != 3:
        # Fallback to LUW for other k
        return _construct_luw_fallback(k, g)

    # Strategy: Search for Circulant Graphs C_n(1, a, n-a)
    # n must be even.
    # Girth g=16.
    
    n_min = 512
    max_n = 1500 # Upper bound for search
    
    # Iterate n
    for n in range(n_min, max_n, 2):
        if time.time() > deadline - 5:
            break
            
        # Iterate k (step size)
        # k must be such that {1, k, n-k} are distinct and no multi-edges
        # k != 1, k != n-1 (same as 1)
        # k != n/2 (multi-edge)
        for a in range(2, n // 2):
            if a == n // 2:
                continue
                
            # Quick check: girth 3
            # Triangle in C_n(1, a, n-a)
            # Steps s1, s2, s3 in {1, -1, a, -a}
            # Sum 0 mod n
            # 1 + 1 - 2 = 0 (if a=2, but a>=2. If a=2, 2 is a step. 1+1-2=0. Triangle.)
            # So if a == 2, girth 3. Skip.
            if a == 2:
                continue
                
            # Check for girth < g
            if _has_girth_at_least_circulant(n, a, g):
                edges = _get_circulant_edges(n, a)
                return n, edges

    # Fallback to LUW if circulant search fails
    for d in range(8, 15):
        if time.time() > deadline - 5:
            break
        try:
            n_luw, edges_luw = _construct_luw_optimized(3, d)
            if _has_girth_at_least_fast(n_luw, edges_luw, g):
                return n_luw, edges_luw
        except Exception:
            continue
            
    # Last resort
    n_luw, edges_luw = _construct_luw_optimized(3, 20)
    return n_luw, edges_luw

def _get_circulant_edges(n, a):
    edges = []
    seen = set()
    for i in range(n):
        for step in [1, a, n-a]:
            j = (i + step) % n
            if i < j:
                if (i, j) not in seen:
                    edges.append((i, j))
                    seen.add((i, j))
    return edges

def _has_girth_at_least_circulant(n, a, g):
    # BFS from 0
    # dist[v] = distance from 0
    # parent[v] = previous vertex
    # If we find a back-edge (u, v) such that dist[u] + dist[v] + 1 < g, return False
    # Note: In undirected graph, every edge is (u, v) and (v, u).
    # BFS tree edges are fine. Non-tree edges form cycles.
    
    dist = [-1] * n
    parent = [-1] * n
    dist[0] = 0
    q = deque([0])
    
    # Precompute neighbors for speed?
    # Neighbors of u: (u+1)%n, (u-1)%n, (u+a)%n, (u-a)%n
    
    while q:
        u = q.popleft()
        for v in [(u+1)%n, (u-1)%n, (u+a)%n, (u-a)%n]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
            else:
                # Check if this edge closes a short cycle
                # We only care if the cycle length < g
                # Cycle length = dist[u] + dist[v] + 1
                # But we must ensure it's not the parent edge
                if v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _has_girth_at_least_fast(n, edges, g):
    # Optimized girth check
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _construct_luw_fallback(k, g):
    min_d = max(3, g // 2 - 2)
    for d in range(min_d, min_d + 10):
        try:
            n_luw, edges_luw = _construct_luw_optimized(k, d)
            if _has_girth_at_least_fast(n_luw, edges_luw, g):
                return n_luw, edges_luw
        except Exception:
            continue
    return _construct_luw_optimized(k, min_d + 10)

def _construct_luw_optimized(k, d):
    if k != 3:
        return _construct_luw_generic(k, d)
        
    q = 3
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

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
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _construct_luw_generic(k, d):
    q = k
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

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
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
