import time
import random
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses voltage lift of Petersen graph for k=3.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    if k == 3:
        return _construct_petersen_lift(g, deadline, start_time)
    else:
        # Fallback for other k, though task specifies k=3
        return _construct_luw(k, g, d=7)

def _construct_petersen_lift(g, deadline, start_time):
    """
    Constructs a girth-g graph by lifting the Petersen graph with voltages in Z_m.
    Uses deterministic voltage assignments (powers of generators) for efficiency.
    """
    # Petersen graph edges (10 vertices, 15 edges)
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), # Outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5), # Inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)  # Spokes
    ]
    
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # Precompute all non-backtracking closed walks of length < g in the Petersen graph
    coeff_vectors = set()
    max_walk_len = g - 1
    
    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0]*15)]
        
        while stack:
            v, prev_v, l, counts = stack.pop()
            
            if l >= max_walk_len:
                continue
            
            for next_v in range(n_base):
                if next_v == prev_v:
                    continue
                if next_v == v:
                    continue
                
                e_idx = edge_index.get((v, next_v))
                if e_idx is None:
                    continue
                
                new_counts = counts.copy()
                u_edge, v_edge = pet_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    new_counts[e_idx] += 1
                else:
                    new_counts[e_idx] -= 1
                
                new_l = l + 1
                if next_v == start_v and new_l > 0:
                    coeff_vectors.add(tuple(new_counts))
                
                stack.append((next_v, v, new_l, new_counts))
    
    # Normalize constraints
    unique_coeffs = set()
    for c in coeff_vectors:
        for i in range(15):
            if c[i] != 0:
                if c[i] < 0:
                    c = tuple(-x for x in c)
                unique_coeffs.add(c)
                break
    coeff_list = list(unique_coeffs)
    
    # Convert to numpy array for fast dot products
    coeff_array = np.array(coeff_list, dtype=np.int64) if coeff_list else np.empty((0, 15), dtype=np.int64)
    
    # Search for m and voltages
    best_n = float('inf')
    best_edges = None
    
    # Generate primes
    primes = []
    candidate = 2
    while len(primes) < 300:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    
    # For each prime m, try deterministic voltage assignments
    for m in primes:
        if time.time() > deadline:
            break
        
        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue
        
        found = False
        
        # Strategy 1: voltages[i] = g^i mod m for various generators g
        for gen in range(2, min(m, 50)):
            if time.time() > deadline:
                break
            
            # Check if gen is a generator (primitive root) - not strictly necessary
            # Just use powers
            voltages = [1] * 15
            power = gen
            for i in range(15):
                voltages[i] = power % m
                power = (power * gen) % m
            
            if _check_voltages(voltages, coeff_array, m):
                edges = _build_lifted_graph(pet_edges, voltages, m, n_base)
                if _has_girth_at_least(n_candidate, edges, g):
                    best_n = n_candidate
                    best_edges = edges
                    found = True
                    return n_candidate, edges
        
        # Strategy 2: voltages based on edge index with offsets
        for offset in range(1, min(m, 20)):
            if time.time() > deadline:
                break
            
            voltages = [(i + offset) % m for i in range(15)]
            if any(v == 0 for v in voltages):
                continue
            
            if _check_voltages(voltages, coeff_array, m):
                edges = _build_lifted_graph(pet_edges, voltages, m, n_base)
                if _has_girth_at_least(n_candidate, edges, g):
                    best_n = n_candidate
                    best_edges = edges
                    found = True
                    return n_candidate, edges
        
        # Strategy 3: Random search (fewer attempts, faster check)
        max_attempts = 200
        for _ in range(max_attempts):
            if time.time() > deadline:
                break
            
            voltages = [random.randint(1, m-1) for _ in range(15)]
            
            if _check_voltages(voltages, coeff_array, m):
                edges = _build_lifted_graph(pet_edges, voltages, m, n_base)
                if _has_girth_at_least(n_candidate, edges, g):
                    best_n = n_candidate
                    best_edges = edges
                    found = True
                    return n_candidate, edges
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback
    return _construct_luw(k=3, g=g, d=11)

def _check_voltages(voltages, coeff_array, m):
    """Check if voltages satisfy all constraints using numpy."""
    if len(coeff_array) == 0:
        return True
    vol = np.array(voltages, dtype=np.int64)
    # Compute all dot products at once
    sums = coeff_array @ vol
    # Check none are 0 mod m
    return not np.any(sums % m == 0)

def _build_lifted_graph(pet_edges, voltages, m, n_base):
    """Build the lifted graph from base graph and voltages."""
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    edges = set()
    for u in range(n_base):
        for v in range(n_base):
            if u == v:
                continue
            e_idx = edge_index.get((u, v))
            if e_idx is None:
                continue
            
            vol = voltages[e_idx]
            u_edge, v_edge = pet_edges[e_idx]
            if u == u_edge and v == v_edge:
                sign = 1
            else:
                sign = -1
            
            for layer in range(m):
                u_node = layer * n_base + u
                target_layer = (layer + sign * vol) % m
                v_node = target_layer * n_base + v
                
                if u_node < v_node:
                    edges.add((u_node, v_node))
                else:
                    edges.add((v_node, u_node))
    
    return list(edges)

def _construct_luw(k, g, d=None, variant=0):
    """
    LUW construction with specified depth d and variant for rule selection.
    For k=3, uses q=3 arithmetic.
    """
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Build rule table with variant-dependent offsets
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
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
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _has_girth_at_least(n, edges, g):
    """
    Check if graph has girth at least g using non-backtracking walks.
    A graph has girth >= g iff there are no non-backtracking closed walks of length < g.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_len = g - 1
    
    for start in range(n):
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
        
        while queue:
            u, prev, l = queue.popleft()
            
            if u == start and l > 0:
                return False
            
            if l >= max_len:
                continue
            
            for v in adj[u]:
                if v != prev:
                    queue.append((v, u, l + 1))
    
    return True
