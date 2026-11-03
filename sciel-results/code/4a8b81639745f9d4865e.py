import time
import sys
from collections import deque
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    if k == 3 and g == 16:
        # Strategy 1: Voltage Lift of Tutte-Coxeter Graph (GQ(2,2) Incidence)
        # Base: 30 vertices, girth 8.
        # Lift: Group Z_31 (order 31). Total vertices 930.
        
        try:
            base_n, base_edges = _construct_tutte_coxeter()
            
            # Find all cycles of length 8, 10, 12, 14 in base graph
            # These are the only cycles that can close in the lift with length < 16
            # if their voltage sum is 0.
            cycles_to_break = []
            for L in [8, 10, 12, 14]:
                cycles_to_break.extend(_find_cycles_of_length(base_n, base_edges, L))
            
            num_edges = len(base_edges)
            group_order = 31
            
            # Random search for voltages
            found = False
            for attempt in range(50000):
                if time.time() > deadline - 20:
                    break
                # Randomly assign voltages (1..30)
                voltages = [random.randint(1, group_order - 1) for _ in range(num_edges)]
                
                # Check constraints: sum != 0 mod 31 for all cycles
                valid = True
                for cycle in cycles_to_break:
                    s = 0
                    for e_idx in cycle:
                        s = (s + voltages[e_idx]) % group_order
                    if s == 0:
                        valid = False
                        break
                
                if valid:
                    # Construct the lifted graph
                    n_lift, edges_lift = _construct_voltage_lift(base_n, base_edges, voltages, group_order)
                    
                    # Verify girth is at least g
                    if _has_girth_at_least_fast(n_lift, edges_lift, g):
                        return n_lift, edges_lift
                    else:
                        # If verification fails, continue searching
                        # This can happen if there are other structures
                        pass
                        
        except Exception:
            pass

        # Strategy 2: LUW Construction (Fallback)
        # For k=3, d=7 gives girth 16 and n=4374
        # This is valid but large.
        rule = _generate_rule_variants(k, 7)[0]
        n_luw, edges_luw = _construct_luw_with_rule(k, 7, rule)
        return n_luw, edges_luw

    # Generic fallback for other k, g
    # Use LUW with appropriate depth
    # Approximate depth needed: g/2 - 1
    d = max(3, (g - 2) // 2)
    rule = _generate_rule_variants(k, d)[0]
    n_luw, edges_luw = _construct_luw_with_rule(k, d, rule)
    return n_luw, edges_luw

def _construct_tutte_coxeter():
    """
    Returns the Tutte-Coxeter Graph (30 vertices, 45 edges, girth 8).
    """
    edges = [
        (0, 15), (0, 16), (0, 17),
        (1, 15), (1, 18), (1, 19),
        (2, 16), (2, 18), (2, 20),
        (3, 17), (3, 19), (3, 21),
        (4, 20), (4, 21), (4, 22),
        (5, 22), (5, 23), (5, 24),
        (6, 23), (6, 24), (6, 25),
        (7, 25), (7, 26), (7, 27),
        (8, 26), (8, 27), (8, 28),
        (9, 28), (9, 29), (9, 15),
        (10, 29), (10, 16), (10, 17),
        (11, 18), (11, 19), (11, 20),
        (12, 21), (12, 22), (12, 23),
        (13, 24), (13, 25), (13, 26),
        (14, 27), (14, 28), (14, 29)
    ]
    return 30, edges

def _find_cycles_of_length(n, edges, length):
    """
    Find all simple cycles of a specific length in a graph.
    Returns list of edge indices.
    """
    adj = [[] for _ in range(n)]
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        adj[u].append(v)
        adj[v].append(u)
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    cycles = []
    
    # DFS from each start node, only counting cycles where start is the smallest node
    for start in range(n):
        stack = [(start, [start], [])]
        while stack:
            node, path, p_edges = stack.pop()
            if len(path) == length:
                if start in adj[node]:
                    if min(path) == start:
                        c_edges = []
                        for i in range(len(path)-1):
                            u, v = path[i], path[i+1]
                            c_edges.append(edge_index[(u, v)])
                        c_edges.append(edge_index[(node, start)])
                        cycles.append(c_edges)
                continue
            
            for neighbor in adj[node]:
                if neighbor == start and len(path) < length:
                    continue
                if neighbor in path:
                    continue
                stack.append((neighbor, path + [neighbor], p_edges))
                
    return cycles

def _construct_voltage_lift(base_n, base_edges, voltages, group_order):
    """
    Construct the voltage lift of the base graph.
    """
    n = base_n * group_order
    edges = []
    
    for i, (u, v) in enumerate(base_edges):
        vol = voltages[i]
        for g in range(group_order):
            u_lift = u * group_order + g
            v_lift = v * group_order + (g + vol) % group_order
            edges.append((u_lift, v_lift))
            
    return n, edges

def _has_girth_at_least_fast(n, edges, g):
    """
    Check if graph has girth at least g using BFS from each vertex.
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
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _generate_rule_variants(k, max_d):
    variants = []
    base_rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, max_d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        base_rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    variants.append(base_rule)
    return variants

def _construct_luw_with_rule(k, d, rule):
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
