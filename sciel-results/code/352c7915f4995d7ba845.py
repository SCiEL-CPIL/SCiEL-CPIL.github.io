import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_n = float('inf')
    best_edges = None
    
    # 1. Deterministic LUW Construction with multiple rule variants
    # For k=3, g=16: Moore bound is 2*(3*(2^14-2)) = 98304... no wait
    # Moore bound for k=3, g=16: n >= 2*(3*(2^(g/2-1)-1)) = 2*(3*(2^7-1)) = 2*3*127 = 762
    # Actually for girth g (even), Moore bound: n >= 2*(k*(2^(g/2-1)-1))
    # For k=3, g=16: n >= 2*(3*(2^7-1)) = 2*3*127 = 762
    # Best known is 936, so we should search around there
    
    min_d = max(3, g // 2 - 2)
    max_d = g + 5  # Safety cap
    
    # Generate multiple rule variants to try
    rule_variants = _generate_rule_variants(k, max_d)
    
    for d in range(min_d, max_d + 1):
        if time.time() > deadline - 10:
            break
        
        for variant_idx, rule in enumerate(rule_variants):
            if time.time() > deadline - 10:
                break
            try:
                n_luw, edges_luw = _construct_luw_with_rule(k, d, rule)
            except Exception:
                continue
            
            if n_luw < best_n:
                # Verify girth using faster method
                if _has_girth_at_least_fast(n_luw, edges_luw, g):
                    best_n = n_luw
                    best_edges = edges_luw
                    # Don't break - try other variants for this d too
                    # But if we found a good one, we can be more selective
                    if best_n <= 1000:
                        break  # Found a reasonably good graph, try next d

    if best_n == float('inf'):
        # Fallback: Return a valid but potentially large graph
        for variant_idx, rule in enumerate(rule_variants):
            try:
                n_luw, edges_luw = _construct_luw_with_rule(k, max_d, rule)
                if _has_girth_at_least_fast(n_luw, edges_luw, g):
                    return n_luw, edges_luw
            except Exception:
                continue
        # Last resort
        n_luw, edges_luw = _construct_luw_with_rule(k, max_d + 3, rule_variants[0])
        return n_luw, edges_luw

    return best_n, best_edges

def _get_petersen_edges():
    edges = []
    for i in range(5):
        edges.append((i, (i+1) % 5))
    inner = [5, 7, 9, 6, 8]
    for i in range(5):
        edges.append((inner[i], inner[(i+1) % 5]))
    for i in range(5):
        edges.append((i, i+5))
    return edges

def _random_lift(base_edges, M):
    n_base = 10
    n_lift = n_base * M
    perm_map = {}
    for u, v in base_edges:
        key = (min(u, v), max(u, v))
        if key not in perm_map:
            p = list(range(M))
            random.shuffle(p)
            perm_map[key] = p
    lift_edges = []
    for u, v in base_edges:
        key = (min(u, v), max(u, v))
        p = perm_map[key]
        for j in range(M):
            u_node = u * M + j
            v_node = v * M + p[j]
            lift_edges.append((u_node, v_node))
    return lift_edges

def _has_girth_at_least_fast(n, edges, g):
    """
    Fast girth verification using non-backtracking walks from each vertex.
    Uses BFS with parent tracking to detect cycles shorter than g.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do a limited BFS to find shortest cycle through it
    # Use non-backtracking: track (node, parent) pairs
    for start in range(n):
        # dist[(u, parent_u)] = distance from start to u, having come from parent_u
        dist = {}
        q = deque()
        for v in adj[start]:
            dist[(start, v)] = 1
            q.append((start, v))
        
        while q:
            (prev, u) = q.popleft()
            d = dist[(prev, u)]
            if d >= g - 1:
                continue  # We only care about cycles < g
            for v in adj[u]:
                if v == prev:
                    continue  # Non-backtracking
                if v == start:
                    # Found a cycle of length d + 1
                    if d + 1 < g:
                        return False
                else:
                    if (u, v) not in dist:
                        dist[(u, v)] = d + 1
                        q.append((u, v))
    return True

def _generate_rule_variants(k, max_d):
    """
    Generate multiple rule variants for LUW construction.
    Returns a list of rule dictionaries.
    """
    if k != 3:
        # For k != 3, use the original rule
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, max_d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return [rule]
    
    # For k=3, generate multiple variants
    variants = []
    
    # Variant 0: Original
    rule0 = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule0[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    variants.append(rule0)
    
    # Variant 1: Different starting pattern
    rule1 = {2: (1, 1), 3: (1, 2), 4: (2, 1)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule1[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    variants.append(rule1)
    
    # Variant 2: Shifted pattern
    rule2 = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # Use different selection
        rule2[c] = [(1, c0 - 2), (c0, 1), (c0 - 1, 1), (1, c0 + 1)][r]
    variants.append(rule2)
    
    # Variant 3: Another shift
    rule3 = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule3[c] = [(1, c0 - 2), (c0 - 1, 1), (1, c0 + 1), (c0, 1)][r]
    variants.append(rule3)
    
    # Variant 4: Another variation
    rule4 = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule4[c] = [(c0 - 1, 1), (1, c0 - 2), (c0, 1), (1, c0 + 1)][r]
    variants.append(rule4)
    
    # Variant 5: Yet another
    rule5 = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule5[c] = [(1, c0 + 1), (1, c0 - 2), (c0 - 1, 1), (c0, 1)][r]
    variants.append(rule5)
    
    return variants

def _construct_luw_with_rule(k, d, rule):
    """
    Construct LUW graph with a specific rule.
    """
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

# Keep old functions for compatibility but they're no longer called from construct
def _construct_luw_optimized(k, d):
    rule = _generate_rule_variants(k, d)[0]
    return _construct_luw_with_rule(k, d, rule)

def _construct_luw_generic(k, d):
    return _construct_luw_optimized(k, d)
# EVOLVE-BLOCK-END
