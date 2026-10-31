import time
import random
import numpy as np
from collections import deque

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_girth_fast(n, edges, min_girth):
    """Fast girth check using BFS from each vertex with parent tracking."""
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2

    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])

        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for 3-regular graphs.
    We try smaller d values to get fewer vertices, checking girth empirically.
    """
    if k != 3:
        # Fallback for other k (should not be hit for this task)
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    # For k=3, we use q=3.
    q = 3

    def build_luw(d):
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

    # Try d=8 first. If girth < 16, try d=9, etc.
    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    # If all fail (unlikely for d=11), return the largest one (which should be valid by theory)
    return build_luw(11)

def _cayley_graph_zm_x_zn(m, n, a1, a2, b1, b2, deadline):
    """
    Construct Cayley graph of Z_m x Z_n with generators:
    g1 = (a1, 0), g2 = (0, a2), g3 = (b1, b2)
    where a1, a2, b1, b2 are chosen so that the generators form a symmetric set
    (i.e., g^-1 is also in the set) and no generator is the identity.
    
    For 3-regular, we need 3 generators. But Cayley graphs with a symmetric generating
    set of size 3 require that the inverses of the generators are also in the set,
    which means we need pairs {g, g^-1}. So we need either:
    - 3 generators where one is its own inverse (order 2) and two others form a pair
    - Or we use a different approach: use generators g1, g2, g3 where g1^-1 = g2, g3^-1 = g3
    """
    # For simplicity, use generators: (a1, 0), (a2, 0) where a1 + a2 = 0 mod m (so they're inverses)
    # and (0, a3) where a3 = -a3 mod n (so a3 has order 2, i.e., n is even and a3 = n//2)
    # This gives a 3-regular Cayley graph.
    
    # Actually, let's use a cleaner approach: generators (1,0), (-1,0), (0,1) in Z_m x Z_n
    # This is 3-regular if we don't double-count. But (1,0) and (-1,0) are distinct unless m=2.
    # And (0,1) is distinct from both unless n=1.
    
    # For 3-regular, we need exactly 3 neighbors per vertex. With generators S = {(a,b), (c,d), (e,f)},
    # we need |S| = 3 and S is symmetric (S = S^-1). This means one generator must be self-inverse.
    
    # Let's use: g1 = (1, 0), g2 = (-1, 0), g3 = (0, 1)
    # g1^-1 = g2, g3^-1 = (0, -1) which is not in S unless n=2.
    # So this doesn't work for n > 2.
    
    # Better: use g1 = (1, 0), g2 = (0, 1), g3 = (-1, -1)
    # g1^-1 = (-1, 0) not in S. Doesn't work.
    
    # For a 3-regular Cayley graph, we need a symmetric generating set of size 3.
    # This requires one element of order 2 and a pair of inverses.
    # So: S = {s, s^-1, t} where t = t^-1 (order 2).
    
    # In Z_m x Z_n, elements of order 2: (m//2, 0) if m even, (0, n//2) if n even, (m//2, n//2) if both even.
    
    # Let's try: m even, n odd.
    # t = (m//2, 0) [order 2]
    # s = (1, 1), s^-1 = (-1, -1)
    # S = {(1,1), (-1,-1), (m//2, 0)}
    
    if m % 2 != 0:
        return None
    
    t = (m // 2, 0)
    s = (1, 1)
    s_inv = ((-1) % m, (-1) % n)
    
    # Check that s != s_inv and s != t and s_inv != t
    if s == s_inv or s == t or s_inv == t:
        return None
    
    generators = [s, s_inv, t]
    
    n_vertices = m * n
    edges = set()
    
    for i in range(m):
        for j in range(n):
            v = i * n + j
            for (di, dj) in generators:
                ni = (i + di) % m
                nj = (j + dj) % n
                u = ni * n + nj
                if v < u:
                    edges.add((v, u))
    
    if len(edges) != n_vertices * 3 // 2:
        return None
    
    return n_vertices, list(edges)

def _cayley_search(k, g, deadline):
    """
    Search for small 3-regular Cayley graphs with girth >= g.
    Use Z_m x Z_n with the generator scheme above.
    """
    if k != 3:
        return None
    
    best = None
    
    # Try various m, n with m even
    for m in range(6, 200, 2):  # m even, at least 6
        if time.time() > deadline:
            break
        for n in range(3, 200):  # n odd preferred, but try all
            if time.time() > deadline:
                break
            
            result = _cayley_graph_zm_x_zn(m, n, 1, 1, m//2, 0, deadline)
            if result is None:
                continue
            
            n_vertices, edges = result
            
            # Quick check: if n_vertices is already larger than best, skip
            if best is not None and n_vertices >= best[0]:
                continue
            
            # Check girth
            if _check_girth_fast(n_vertices, edges, g):
                if best is None or n_vertices < best[0]:
                    best = (n_vertices, edges)
    
    return best

def _deterministic_voltage_lift(k, g, deadline, seed=42):
    """
    Voltage lift of the Petersen graph using a deterministic, algebraic voltage
    assignment over a cyclic group of order m.
    """
    if k != 3:
        return None
    
    random.seed(seed)
    
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    m_edges = len(petersen_edges)  # 15
    
    # Build adjacency with edge indices
    adj_base = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(petersen_edges):
        adj_base[u].append((v, i))
        adj_base[v].append((u, i))
    
    # Find all simple cycles in Petersen graph
    cycles = []
    seen = set()
    
    def dfs_cycles(start, curr, visited, edge_path, length):
        for next_v, edge_idx in adj_base[curr]:
            if next_v == start and length >= 3:
                cycle_key = tuple(sorted(edge_path))
                if cycle_key not in seen:
                    seen.add(cycle_key)
                    cycles.append(cycle_key)
            elif next_v not in visited:
                visited.add(next_v)
                dfs_cycles(start, next_v, visited, edge_path + [edge_idx], length + 1)
                visited.remove(next_v)
    
    for start in range(n_base):
        visited = {start}
        dfs_cycles(start, start, visited, [], 0)
    
    # Try different moduli m
    primes_to_try = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    for m in primes_to_try:
        if time.time() > deadline:
            break
        
        best_result = None
        for a in range(1, m):
            for b in range(m):
                voltages = [(a * i + b) % m for i in range(m_edges)]
                
                # Check all cycles
                valid = True
                for C in cycles:
                    s = sum(voltages[i] for i in C) % m
                    if s == 0:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Build the lift
                n = 10 * m
                edges = []
                for idx, (u, v) in enumerate(petersen_edges):
                    vol = voltages[idx]
                    for j in range(m):
                        a_v = u * m + j
                        b_v = v * m + ((j + vol) % m)
                        edges.append((a_v, b_v))
                
                # Verify girth
                if _check_girth_fast(n, edges, g):
                    return n, edges
                
                # Keep track of the best result (smallest n)
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
        
        # If no deterministic assignment works for this m, try random as a fallback
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            voltages = [random.randint(1, m - 1) for _ in range(m_edges)]
            
            # Check all cycles
            valid = True
            for C in cycles:
                s = sum(voltages[i] for i in C) % m
                if s == 0:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Build the lift
            n = 10 * m
            edges = []
            for idx, (u, v) in enumerate(petersen_edges):
                vol = voltages[idx]
                for j in range(m):
                    a_v = u * m + j
                    b_v = v * m + ((j + vol) % m)
                    edges.append((a_v, b_v))
            
            # Verify girth
            if _check_girth_fast(n, edges, g):
                return n, edges
    
    return None

def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # 10s margin

    if k != 3:
        return _luw_construct(k, g)

    # Try Cayley graph search first (structural change)
    result = _cayley_search(k, g, deadline - 30)
    if result is not None:
        n, edges = result
        if _check_girth_fast(n, edges, g):
            return n, edges

    # Try deterministic voltage lift
    result = _deterministic_voltage_lift(k, g, deadline - 10, seed=42)
    if result is not None:
        n, edges = result
        if _check_girth_fast(n, edges, g):
            return n, edges

    # Fallback to LUW
    return _luw_construct(k, g)
