import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    if k == 3 and g == 16:
        best_n = float('inf')
        best_edges = None
        
        # Try multiple rule variants for each d, smallest d first
        # For LUW(3,d): n = 2 * 3^d
        # d=7: n=4374, d=8: n=13122, d=9: n=39366
        # We want the smallest d that gives girth >= 16
        
        rule_variants = _generate_rule_variants()
        
        for d in [7, 8, 9, 10]:
            if time.time() > deadline - 30:
                break
            for rv_idx, rule in enumerate(rule_variants):
                if time.time() > deadline - 10:
                    break
                try:
                    n_luw, edges_luw = _construct_luw_with_rule(k, d, rule)
                    if n_luw >= best_n:
                        continue
                    # Fast girth check with early termination
                    if _has_girth_at_least_fast(n_luw, edges_luw, g):
                        best_n = n_luw
                        best_edges = edges_luw
                        break  # Found valid for this d, move to next d
                except Exception:
                    continue
        
        if best_n != float('inf'):
            return best_n, best_edges
            
    # Generic fallback
    min_d = max(3, g // 2 - 2)
    max_d = g + 5
    
    for d in range(min_d, max_d + 1):
        if time.time() > deadline - 15:
            break
        try:
            n_luw, edges_luw = _construct_luw_optimized(k, d)
            if _has_girth_at_least_fast(n_luw, edges_luw, g):
                return n_luw, edges_luw
        except Exception:
            continue
                
    # Last resort - use a larger d to ensure validity
    n_luw, edges_luw = _construct_luw_optimized(k, max_d + 3)
    return n_luw, edges_luw

def _generate_rule_variants():
    """
    Generate multiple LUW rule variants for k=3.
    Each variant is a dict mapping c -> (a, b).
    """
    variants = []
    
    # Variant 0: Standard cyclic pattern (current)
    def make_rule_pattern(d, pattern_type):
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            if pattern_type == 0:  # Standard 4-cycle
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
            elif pattern_type == 1:  # Shifted pattern
                r = (c - 4) % 4
                c0 = c - r
                if r == 0:
                    rule[c] = (1, c0 - 1)
                elif r == 1:
                    rule[c] = (c0 - 2, 1)
                elif r == 2:
                    rule[c] = (c0 - 1, 1)
                else:
                    rule[c] = (1, c0)
            elif pattern_type == 2:  # Reverse pattern
                r = (c - 5) % 4
                c0 = c - r
                if r == 0:
                    rule[c] = (c0 - 2, 1)
                elif r == 1:
                    rule[c] = (1, c0 - 1)
                elif r == 2:
                    rule[c] = (1, c0)
                else:
                    rule[c] = (c0 + 1, 1)
            elif pattern_type == 3:  # Diagonal pattern
                rule[c] = (min(c-1, d), min(c-2, d))
                if rule[c][1] < 1:
                    rule[c] = (1, 1)
            elif pattern_type == 4:  # Alternating
                if c % 2 == 0:
                    rule[c] = (1, c - 1)
                else:
                    rule[c] = (c - 1, 1)
                if rule[c][1] > d:
                    rule[c] = (1, d)
            elif pattern_type == 5:  # Fibonacci-like
                rule[c] = (max(1, c - 2), max(1, c - 1))
                if rule[c][1] > d:
                    rule[c] = (1, d)
        return rule
    
    for pt in range(6):
        variants.append(pt)
    
    return variants

def _construct_luw_with_rule(k, d, pattern_type):
    """
    LUW construction with a specific rule pattern.
    """
    q = k
    
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        if pattern_type == 0:
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
        elif pattern_type == 1:
            r = (c - 4) % 4
            c0 = c - r
            if r == 0:
                rule[c] = (1, c0 - 1)
            elif r == 1:
                rule[c] = (c0 - 2, 1)
            elif r == 2:
                rule[c] = (c0 - 1, 1)
            else:
                rule[c] = (1, c0)
        elif pattern_type == 2:
            r = (c - 5) % 4
            c0 = c - r
            if r == 0:
                rule[c] = (c0 - 2, 1)
            elif r == 1:
                rule[c] = (1, c0 - 1)
            elif r == 2:
                rule[c] = (1, c0)
            else:
                rule[c] = (c0 + 1, 1)
        elif pattern_type == 3:
            a = min(c-1, d)
            b = min(c-2, d)
            if b < 1: b = 1
            if a < 1: a = 1
            rule[c] = (a, b)
        elif pattern_type == 4:
            if c % 2 == 0:
                rule[c] = (1, min(c - 1, d))
            else:
                rule[c] = (min(c - 1, d), 1)
        elif pattern_type == 5:
            a = max(1, min(c - 2, d))
            b = max(1, min(c - 1, d))
            rule[c] = (a, b)

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

def _has_girth_at_least_fast(n, edges, g):
    """
    Check if graph has girth at least g using BFS from each vertex.
    Returns True if no cycle shorter than g exists.
    Optimized with early termination and array-based adjacency.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For 3-regular graphs, girth >= g means no cycle of length < g
    # Use BFS with depth limit
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            du = dist[u]
            if du + 1 >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _construct_luw_optimized(k, d):
    """
    Optimized LUW construction for 3-regular graphs.
    Uses the standard LUW rules for k=3.
    """
    return _construct_luw_with_rule(k, d, 0)
# EVOLVE-BLOCK-END
