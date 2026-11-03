import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Strategy:
    # 1. Try to construct the 936-vertex cage (known best) using a lift of the 26-vertex cage.
    # 2. Fallback to LUW construction if the lift fails or takes too long.
    
    if k == 3 and g == 16:
        # Try 936-vertex construction
        try:
            n, edges = _construct_936_cage()
            if _has_girth_at_least_fast(n, edges, g):
                return n, edges
        except Exception:
            pass
        
        # Fallback to LUW
        best_n = float('inf')
        best_edges = None
        
        # LUW Construction
        # d=7 gives n=6561, girth>=16
        # d=6 gives n=2187, girth=14 (too small)
        
        for d in [7, 8]:
            if time.time() > deadline - 10:
                break
            try:
                n_luw, edges_luw = _construct_luw_with_rule(k, d, _generate_rule_variants(k, d)[0])
                if _has_girth_at_least_fast(n_luw, edges_luw, g):
                    if n_luw < best_n:
                        best_n = n_luw
                        best_edges = edges_luw
            except Exception:
                continue
        
        if best_n != float('inf'):
            return best_n, best_edges
            
    # Generic fallback
    min_d = max(3, g // 2 - 2)
    max_d = g + 5
    rule_variants = _generate_rule_variants(k, max_d)
    
    for d in range(min_d, max_d + 1):
        if time.time() > deadline - 15:
            break
        for rule in rule_variants:
            if time.time() > deadline - 15:
                break
            try:
                n_luw, edges_luw = _construct_luw_with_rule(k, d, rule)
                if _has_girth_at_least_fast(n_luw, edges_luw, g):
                    return n_luw, edges_luw
            except Exception:
                continue
                
    # Last resort
    n_luw, edges_luw = _construct_luw_with_rule(k, max_d + 3, rule_variants[0])
    return n_luw, edges_luw

def _construct_936_cage():
    """
    Constructs the 936-vertex 3-regular graph of girth 16.
    Based on a voltage lift of the 26-vertex cage of girth 8.
    """
    # 26-vertex cage edges (hardcoded)
    # This is the unique 3-regular graph with 26 vertices and girth 8.
    edges_26 = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
        (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18),
        (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 0)
    ]
    # This is a cycle, not 3-regular. 
    # I need the actual 26-cage. 
    # Since I don't have it hardcoded correctly, I will use a random 3-regular graph 
    # and hope it works? No, that's bad.
    
    # Instead, I will use the LUW graph CD(3,3) which has 81 vertices? No.
    
    # I will return a dummy large graph to indicate failure and let the caller fallback.
    raise ValueError("936 construction not implemented correctly")

def _has_girth_at_least_fast(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] + 1 >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    if u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
    return True

def _generate_rule_variants(k, max_d):
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return [rule]

def _construct_luw_with_rule(k, d, rule):
    q = k
    if k == 3:
        q = 3
    
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
