import time
import random
from collections import deque
import numpy as np

def _validate_girth(n, edges, g, degree=4):
    """
    Checks if the graph with n vertices and given edges has girth at least g.
    Uses non-backtracking walks from each vertex with early termination.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != degree:
            return False

    # Girth check using BFS on (node, prev) state space
    # Key optimization: if we find ANY cycle of length < g, return False immediately
    for start in range(n):
        # dist[(curr, prev)] = depth
        dq = deque()
        dist = {}
        
        # Start from all neighbors of start
        for nb in adj[start]:
            state = (nb, start)
            dq.append(state)
            dist[state] = 1
            
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            
            # If we reached start, we found a cycle
            if curr == start:
                cycle_len = depth + 1
                if cycle_len < g:
                    return False
                continue

            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
                    
    return True

def _random_4_regular(n, rng):
    """
    Generates a random 4-regular simple graph on n vertices.
    Returns list of edges or None.
    """
    if n < 4:
        return None
    
    # Configuration Model with retry
    for _ in range(10):
        stubs = []
        for v in range(n):
            for _ in range(4):
                stubs.append(v)
            
        edges = set()
        valid = True
        
        rng.shuffle(stubs)
        
        for i in range(0, n * 4, 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                valid = False
                break
            edge = (min(u, v), max(u, v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
        
        if valid:
            return list(edges)
    
    return None

def _algebraic_4_regular(n, rng):
    """
    Generates a structured 4-regular graph using finite field arithmetic.
    This is more likely to have high girth than pure random.
    """
    if n < 4:
        return None
    
    # Try to use a finite field structure
    # For n = p where p is prime, use GF(p)
    # For composite n, use CRT or other structures
    
    # Simple approach: use a circulant-like structure with algebraic offsets
    # Each vertex i connects to i + a_j mod n for j in {1,2,3,4}
    # where a_j are chosen to avoid short cycles
    
    # Better: use a voltage graph lift
    # Base graph: 4 vertices in a cycle (C4)
    # Lift by voltage group Z_m where m = n/4
    
    if n % 4 != 0:
        return None
    
    m = n // 4
    
    # Voltage assignment: each edge in C4 gets a voltage in Z_m
    # The lifted graph has n vertices
    # Vertex (i, j) where i in {0,1,2,3}, j in Z_m
    # Edges: (i, j) -- (i+1 mod 4, j + v_i mod m)
    
    # Choose voltages to maximize girth
    # For girth 10, we want no cycles of length 4, 6, or 8
    
    # Try random voltages
    for _ in range(20):
        voltages = [rng.randint(1, m-1) for _ in range(4)]
        
        # Check if this gives a simple graph (no multi-edges)
        # and compute girth
        edges = set()
        valid = True
        
        for i in range(4):
            for j in range(m):
                u = i * m + j
                v_idx = (i + 1) % 4
                v = v_idx * m + (j + voltages[i]) % m
                edge = (min(u, v), max(u, v))
                if edge in edges:
                    valid = False
                    break
                edges.add(edge)
                # Also add the reverse direction edge
                u2 = v_idx * m + (j - voltages[i]) % m
                v2 = i * m + j
                edge2 = (min(u2, v2), max(u2, v2))
                if edge2 in edges:
                    valid = False
                    break
                edges.add(edge2)
        
        if not valid:
            continue
        
        edge_list = list(edges)
        
        # Quick girth check
        if _validate_girth(n, edge_list, 10, degree=4):
            return edge_list
    
    return None

def construct(k, g):
    t_start = time.time()
    deadline = t_start + 110.0 # 10s margin
    
    # 1. Generate LUW baseline
    # The current program's LUW logic
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
                
    n_luw = len(order)
    
    # If k != 4 or g != 10, just return LUW (or whatever it is)
    # The task is specifically for k=4, g=10?
    # "The evaluator passes this task's degree k and girth g."
    # But the prompt title is "Small 4-regular graph of girth 10".
    # We assume k=4, g=10 is the target case.
    
    if k != 4 or g != 10:
        return n_luw, edges_luw
        
    # 2. Try to find smaller graph
    # Lower bound for (4,10) is 243.
    # Known best is 320.
    # Try n from 243 to 319.
    
    rng = random.Random(42)
    
    best_n = n_luw
    best_edges = edges_luw
    
    # Time budget for search
    search_deadline = t_start + 100.0
    
    # Strategy:
    # 1. First try algebraic constructions for n divisible by 4
    # 2. Then try random graphs
    
    # Phase 1: Algebraic constructions
    for n in range(244, min(best_n, 320), 4):
        if time.time() > search_deadline:
            break
        edges = _algebraic_4_regular(n, rng)
        if edges is not None:
            best_n = n
            best_edges = edges
    
    # Phase 2: Random search for remaining time
    while time.time() < search_deadline:
        n = rng.randint(243, best_n - 1)
        
        # Try algebraic first if n divisible by 4
        if n % 4 == 0:
            edges = _algebraic_4_regular(n, rng)
            if edges is not None:
                best_n = n
                best_edges = edges
                continue
        
        # Try random
        edges = _random_4_regular(n, rng)
        if edges is None:
            continue
            
        # Validate girth
        if _validate_girth(n, edges, 10, degree=4):
            best_n = n
            best_edges = edges
    
    return best_n, best_edges
