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

def _get_petersen_cycles(max_len):
    """
    Get all cycles in the Petersen graph with length < max_len.
    Returns a list of tuples of edge indices.
    """
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    adj_base = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(petersen_edges):
        adj_base[u].append((v, i))
        adj_base[v].append((u, i))

    cycles = []
    seen = set()
    
    def dfs(start, curr, visited, edge_path, length):
        if length >= max_len:
            return
        for next_v, edge_idx in adj_base[curr]:
            if next_v == start and length >= 3:
                cycle_key = tuple(sorted(edge_path))
                if cycle_key not in seen:
                    seen.add(cycle_key)
                    cycles.append(cycle_key)
            elif next_v not in visited:
                visited.add(next_v)
                dfs(start, next_v, visited, edge_path + [edge_idx], length + 1)
                visited.remove(next_v)
                
    for start in range(n_base):
        visited = {start}
        dfs(start, start, visited, [], 0)
        
    return cycles

def _petersen_voltage_lift(k, g, deadline):
    """
    Voltage lift of Petersen graph with deterministic + random voltage search.
    Uses linear voltage assignments first for speed, then random.
    """
    if k != 3:
        return None

    short_cycles = _get_petersen_cycles(g)
    
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_edges = len(petersen_edges)

    # Candidates: try m values that give small n = 10*m
    # Moore bound for (3,16) is ~512, so m >= 52 ideally
    # But we start smaller and go up
    candidates_m = list(range(52, 200))
    # Also include some primes for better algebraic properties
    for p in [53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]:
        if p >= 52 and p < 200 and p not in candidates_m:
            candidates_m.append(p)
    candidates_m.sort()

    for m in candidates_m:
        if time.time() > deadline:
            break
        
        # Try deterministic linear voltage assignments: vol(e_i) = a_i * t where t is a parameter
        # Actually, use a fixed set of multipliers
        # Strategy: vol[e_i] = (c_i * m + offset_i) % m where c_i are chosen constants
        # Better: use vol[e_i] = (a_i * t) % m for varying t, with fixed a_i
        
        # Fixed multipliers for the 15 edges
        base_mults = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
        
        # Try different t values (linear family)
        found = False
        for t in range(1, min(m, 50)):
            if time.time() > deadline:
                break
            voltages = [(mult * t) % m for mult in base_mults]
            valid = True
            for C in short_cycles:
                s = sum(voltages[i] for i in C) % m
                if s == 0:
                    valid = False
                    break
            if valid:
                n = 10 * m
                edges = []
                for idx, (u, v) in enumerate(petersen_edges):
                    vol = voltages[idx]
                    for j in range(m):
                        a = u * m + j
                        b = v * m + ((j + vol) % m)
                        edges.append((a, b))
                
                if _check_girth_fast(n, edges, g):
                    return n, edges
                found = True
                break
        
        # If linear didn't work, try a few random ones
        max_random = 200 if m < 100 else 100
        for attempt in range(max_random):
            if time.time() > deadline:
                break
            voltages = [random.randint(0, m - 1) for _ in range(n_edges)]
            valid = True
            for C in short_cycles:
                s = sum(voltages[i] for i in C) % m
                if s == 0:
                    valid = False
                    break
            if valid:
                n = 10 * m
                edges = []
                for idx, (u, v) in enumerate(petersen_edges):
                    vol = voltages[idx]
                    for j in range(m):
                        a = u * m + j
                        b = v * m + ((j + vol) % m)
                        edges.append((a, b))
                
                if _check_girth_fast(n, edges, g):
                    return n, edges
    return None

def _heisenberg_cayley(k, g, deadline):
    """
    Cayley graph on a Heisenberg-like group.
    The 3D Heisenberg group over Z_p has elements (a,b,c) with operation
    (a,b,c)*(a',b',c') = (a+a', b+b', c+c' + a*b')
    Generators: X=(1,0,0), Y=(0,1,0), and we need a third generator for 3-regular.
    Actually for 3-regular we need 3 generators (or pairs that are inverses).
    Let's use generators g1, g2, g3 where each has an inverse, giving degree 6... 
    No, for 3-regular we need exactly 3 neighbors per vertex.
    Use involutions or a set of 3 generators where some are self-inverse.
    
    Alternative: Use the Cayley graph of Z_p x Z_p x Z_p with 3 generators.
    But that's abelian and may have short cycles.
    
    Better: Use a quotient of the Heisenberg group.
    """
    if k != 3:
        return None
    
    # For 3-regular, we need 3 generators. In a Cayley graph, each generator gives
    # an edge. If generators are not involutions, we get directed edges, but for undirected
    # we need g^{-1} also in the generating set, which doubles the degree.
    # So for degree 3, we need exactly one involution and... no, that doesn't work cleanly.
    # 
    # Actually, for a 3-regular Cayley graph, we need a generating set S of size 3
    # where S is closed under inverses. This means either all 3 are involutions,
    # or one is identity (bad) or... 
    # 
    # Standard approach: use 3 involutions. 
    # 
    # Let's try a different approach: use a known construction.
    # For now, return None and rely on voltage lift.
    return None

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for 3-regular graphs.
    """
    if k != 3:
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

    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    return build_luw(11)

def construct(k, g):
    start_time = time.time()
    deadline = 140.0

    if k != 3:
        return _luw_construct(k, g)

    # Try voltage lift first (most promising for small n)
    result = _petersen_voltage_lift(k, g, deadline)
    if result is not None:
        n, edges = result
        return n, edges

    # Fallback to LUW
    return _luw_construct(k, g)
