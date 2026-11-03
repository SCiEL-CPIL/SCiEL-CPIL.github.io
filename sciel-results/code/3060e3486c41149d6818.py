import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    best_n = float('inf')
    best_edges = None
    
    if k != 3:
        # Fallback for other k
        return _construct_luw_generic(k, g)
        
    # 1. Try Cyclic Lift of Petersen Graph
    # Moore bound for k=3, g=16 is 512.
    # Petersen has 10 vertices.
    # Try M such that 10*M >= 512 => M >= 52.
    petersen_edges = _get_petersen_edges()
    
    # Try M from 52 upwards
    for M in range(52, 60):
        if time.time() > deadline - 30:
            break
        n = 10 * M
        if n >= best_n:
            continue
            
        # Search for cyclic voltages
        found = False
        for attempt in range(200):
            if time.time() > deadline - 30:
                break
            voltages = [random.randint(0, M - 1) for _ in range(len(petersen_edges))]
            edges = _cyclic_lift(petersen_edges, M, voltages)
            if _has_girth_at_least_fast(n, edges, g):
                best_n = n
                best_edges = edges
                found = True
                break
                
        if found:
            return best_n, best_edges

    # 2. Fallback to LUW
    min_d = max(3, g // 2 - 2)
    max_d = g + 5
    
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
                if _has_girth_at_least_fast(n_luw, edges_luw, g):
                    best_n = n_luw
                    best_edges = edges_luw
                    break

    if best_n == float('inf'):
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

def _cyclic_lift(base_edges, M, voltages):
    n_base = 10
    n_lift = n_base * M
    lift_edges = []
    for i, (u, v) in enumerate(base_edges):
        vol = voltages[i]
        for j in range(M):
            u_node = u * M + j
            v_node = v * M + (j + vol) % M
            lift_edges.append((u_node, v_node))
    return lift_edges

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
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
                if dist[u] + 1 >= g:
                    break
    return True

def _generate_rule_variants(k, max_d):
    if k != 3:
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, max_d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return [rule]
    
    variants = []
    rule0 = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule0[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    variants.append(rule0)
    
    rule1 = {2: (1, 1), 3: (1, 2), 4: (2, 1)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule1[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    variants.append(rule1)
    
    return variants

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

def _construct_luw_generic(k, g):
    d = max(3, g // 2 - 2)
    rule = _generate_rule_variants(k, d + 5)[0]
    return _construct_luw_with_rule(k, d, rule)
# EVOLVE-BLOCK-END
