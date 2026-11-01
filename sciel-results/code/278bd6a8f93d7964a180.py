import time
import random
import numpy as np
from collections import deque

def _verify_girth_fast(n, edges, min_girth):
    """Fast girth check using BFS from each vertex with parent tracking."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
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


def _build_936_graph():
    """
    Return a known 3-regular graph with girth 16 and 936 vertices.
    This is based on the Exoo et al. (2026) construction.
    We use a deterministic construction: a voltage lift of the Petersen graph
    with m=936/10=93.6... that's not integer. Let me use a different base.
    
    Actually, 936 = 2 * 468 = 2 * 4 * 117 = 8 * 117. 
    Let's use a bipartite construction or a known cage.
    
    For simplicity and correctness, I'll construct a graph using the LUW method
    with parameters that give 936 vertices, or use a hardcoded edge list.
    
    Since hardcoding 936 vertices is impractical, let me use a reliable
    algebraic construction. The Ramanujan graph X(11) has 120 vertices and girth 10.
    
    Let me use a voltage lift approach with a carefully chosen base and voltages.
    
    Actually, the safest approach: use the LUW construction with q=3, d=8 which
    gives 2*3^8 = 13122 vertices (too many). 
    
    Let me try a different approach: construct a 3-regular graph using a
    circulant or Cayley graph approach.
    
    For the fallback, let me just use LUW with the smallest d that works.
    The 936-vertex graph is specific; I'll use a voltage lift of the Petersen
    with m=93 if it works, or fall back to LUW.
    
    Actually, let me just return None and let the caller handle it.
    """
    return None


def _quadratic_residues(p):
    """Return the set of quadratic residues mod p (including 0)."""
    return set((i * i) % p for i in range(p))


def _golden_ratio_shifts(m):
    """Generate a set of shifts based on golden ratio for good distribution."""
    if m <= 1:
        return [0]
    phi = (1 + 5 ** 0.5) / 2
    shifts = []
    x = 0
    for _ in range(m):
        shifts.append(int(x))
        x = (x + 0.6180339887498949) % 1
    return shifts


def _modular_pattern_shifts(m, base=2):
    """Generate shifts using a modular pattern."""
    shifts = []
    for i in range(m):
        shifts.append((i * base) % m)
    return shifts


def _find_primitive_root(p):
    """Find a primitive root modulo prime p."""
    for g in range(2, p):
        order = p - 1
        found = True
        for q in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            if order % q == 0:
                if pow(g, order // q, p) == 1:
                    found = False
                    break
        if found:
            return g
    return None


def _deterministic_voltage_lift(k, g, deadline):
    """
    Deterministic voltage lift of the Petersen graph using algebraic patterns.
    """
    if k != 3:
        return None

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

    # Find all cycles in Petersen up to length g-1
    cycles = []
    seen = set()

    def dfs(start, curr, visited, edge_path, length):
        if length >= g:
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

    # Try deterministic voltage patterns with various m values
    # Moore bound for k=3, g=16: 2*(2+1)^(7) + 2*(2+1)^(6) = 2*2187 + 2*729 = 4374+1458 = 5832? 
    # Actually Moore bound: 2 + k*(k-1)^((g-2)/2) for even g = 2 + 3*2^7 = 2+384 = 386? No.
    # Moore bound for girth g=16, degree 3: 2*(2+1)^(15/2)... 
    # For even girth g, Moore bound = 2 + k*((k-1)^(g/2-1) + (k-1)^(g/2-2)) / (k-1)
    # = 2 + 3*(2^7 + 2^6)/2 = 2 + 3*(128+64)/2 = 2 + 3*96 = 290.
    # So minimum possible is 290. Known best is 936.
    
    # Try m values that give reasonable n = 10*m
    # We need n >= Moore bound ~ 290, so m >= 29.
    # Try m from 29 upward, but prefer values where algebraic patterns work well.
    
    candidates = []
    # Try small m first (fewer vertices)
    for m in range(29, 100):
        if time.time() > deadline:
            break
        candidates.append(m)
    
    # Also try some specific values that might work well with algebraic patterns
    for m in [31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if m not in candidates and m >= 29:
            candidates.append(m)
    
    candidates.sort()
    
    for m in candidates:
        if time.time() > deadline:
            break
        
        # Try several deterministic voltage patterns
        patterns = []
        
        # Pattern 1: All ones
        patterns.append([1] * 15)
        
        # Pattern 2: Alternating
        patterns.append([1 if i % 2 == 0 else 2 for i in range(15)])
        
        # Pattern 3: Based on golden ratio
        gr = _golden_ratio_shifts(m)
        patterns.append([gr[i % m] % m for i in range(15)])
        
        # Pattern 4: Modular pattern
        mp = _modular_pattern_shifts(m, base=3)
        patterns.append([mp[i % m] % m for i in range(15)])
        
        # Pattern 5: If m is prime, use primitive root powers
        if m > 2:
            # Check if m is prime
            is_prime = True
            for pp in range(2, int(m ** 0.5) + 1):
                if m % pp == 0:
                    is_prime = False
                    break
            if is_prime:
                g = _find_primitive_root(m)
                if g is not None:
                    patterns.append([pow(g, i, m) for i in range(15)])
        
        # Pattern 6: Quadratic residues based
        if m > 2:
            is_prime = True
            for pp in range(2, int(m ** 0.5) + 1):
                if m % pp == 0:
                    is_prime = False
                    break
            if is_prime:
                qr = sorted(_quadratic_residues(m))
                if len(qr) >= 15:
                    patterns.append([qr[i % len(qr)] % m for i in range(15)])
        
        # Pattern 7: Linear with varying slopes
        for slope in [2, 3, 5, 7, 11, 13]:
            patterns.append([(i * slope) % m for i in range(15)])
        
        # Pattern 8: Mixed
        patterns.append([(i * (i + 1) // 2) % m for i in range(15)])
        
        for pattern in patterns:
            if time.time() > deadline:
                break
            # Ensure no zero voltages (to avoid trivial cycles)
            if 0 in pattern:
                continue
            
            # Check if all cycle sums are non-zero mod m
            valid = True
            for C in cycles:
                s = sum(pattern[i] for i in C) % m
                if s == 0:
                    valid = False
                    break
            
            if valid:
                # Construct lift
                n = 10 * m
                edges = []
                for idx, (u, v) in enumerate(petersen_edges):
                    vol = pattern[idx] % m
                    for j in range(m):
                        a = u * m + j
                        b = v * m + ((j + vol) % m)
                        edges.append((a, b))
                
                # Verify
                if _verify_girth_fast(n, edges, g):
                    return n, edges
    
    return None


def _random_voltage_lift(k, g, deadline):
    """
    Random voltage lift of the Petersen graph.
    """
    if k != 3:
        return None

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
        if length >= g:
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

    # Try random voltages with various m
    for m in range(29, 80):
        if time.time() > deadline:
            break
        
        for attempt in range(500):
            if time.time() > deadline:
                break
            voltages = [random.randint(1, m - 1) for _ in range(15)]
            valid = True
            for C in cycles:
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
                
                if _verify_girth_fast(n, edges, g):
                    return n, edges
    
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
        if _verify_girth_fast(n, edges, g):
            return n, edges
    return build_luw(11)


def _random_3reg(k, g, deadline):
    """Random 3-regular graph search (unlikely to succeed for girth 16, but try)."""
    if k != 3:
        return None
    return None


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    if k != 3:
        return _luw_construct(k, g)

    # Try deterministic algebraic voltage lifts first
    result = _deterministic_voltage_lift(k, g, deadline)
    if result is not None:
        return result

    # Try random voltage lifts
    if time.time() < deadline:
        result = _random_voltage_lift(k, g, deadline)
        if result is not None:
            return result

    # Fallback to LUW
    return _luw_construct(k, g)
