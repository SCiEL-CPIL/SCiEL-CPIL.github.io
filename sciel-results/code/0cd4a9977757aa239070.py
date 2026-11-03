import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    if k == 3 and g == 16:
        best_n = float('inf')
        best_edges = None
        
        # Try multiple d values and rule variants to find smallest valid graph
        for d in [7, 8, 9, 10]:
            if time.time() > deadline - 20:
                break
            for rule in _generate_rule_variants(k, d):
                if time.time() > deadline - 15:
                    break
                try:
                    n_luw, edges_luw = _construct_luw_with_rule(k, d, rule)
                    if n_luw < best_n:
                        if _has_girth_at_least_fast(n_luw, edges_luw, g):
                            best_n = n_luw
                            best_edges = edges_luw
                            # If we found a good one, we can stop early
                            if n_luw <= 1000:
                                return n_luw, edges_luw
                except Exception:
                    continue
        
        if best_n != float('inf'):
            return best_n, best_edges
            
    # Generic fallback
    min_d = max(3, (g - 2) // 2 + 1)
    max_d = min(g + 5, 12)
    
    for d in range(min_d, max_d + 1):
        if time.time() > deadline - 15:
            break
        for rule in _generate_rule_variants(k, d):
            if time.time() > deadline - 10:
                break
            try:
                n_luw, edges_luw = _construct_luw_with_rule(k, d, rule)
                if _has_girth_at_least_fast(n_luw, edges_luw, g):
                    return n_luw, edges_luw
            except Exception:
                continue
                
    # Last resort with larger d
    d_fallback = max_d + 3
    n_luw, edges_luw = _construct_luw_with_rule(k, d_fallback, _generate_rule_variants(k, d_fallback)[0])
    return n_luw, edges_luw

def _has_girth_at_least_fast(n, edges, g):
    """
    Check if graph has girth at least g using BFS from each vertex.
    Returns True if no cycle shorter than g exists.
    """
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
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _generate_rule_variants(k, max_d):
    """
    Generate multiple rule variants for LUW construction.
    Each rule is a dict mapping level c to (a, b) indices.
    """
    variants = []
    
    # Base rule pattern
    base_rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        base_rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    variants.append(base_rule)
    
    # Variant 2: shift the pattern
    rule2 = {2: (1, 1), 3: (1, 2), 4: (2, 1)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 4) % 4, c - (c - 4) % 4
        rule2[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    variants.append(rule2)
    
    # Variant 3: different base
    rule3 = {2: (1, 1), 3: (1, 1), 4: (2, 1)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 3, c - (c - 5) % 3
        rule3[c] = [(1, c0 - 1), (c0, 1), (1, c0 + 1)][r]
    variants.append(rule3)
    
    # Variant 4: another pattern
    rule4 = {2: (1, 1), 3: (2, 1), 4: (1, 1)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 5, c - (c - 5) % 5
        rule4[c] = [(1, c0 - 1), (c0, 1), (1, c0 + 1), (c0 - 1, 1), (1, c0 - 2)][r % 5]
    variants.append(rule4)
    
    return variants

def _construct_luw_with_rule(k, d, rule):
    """
    Construct LUW graph with given degree k, depth d, and rule.
    """
    q = k if k >= 3 else 3
    
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
