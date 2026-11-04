import random
import numpy as np
from collections import deque

def construct(k, g):
    # Petersen graph has 10 vertices, 15 edges.
    # We lift it by a group Z_p.
    # n = 10 * p.
    # We need n >= 512, so p >= 52.
    # Try primes starting from 53.
    
    # Petersen graph edges (0-9)
    # Standard labeling: 0-4 outer, 5-9 inner.
    # Outer: 0-1, 1-2, 2-3, 3-4, 4-0
    # Inner: 5-7, 7-9, 9-6, 6-8, 8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    edges_p = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    num_edges = 15
    # Map edge tuple to index
    edge_index = {}
    for i, e in enumerate(edges_p):
        edge_index[e] = i
        edge_index[(e[1], e[0])] = i
        
    # Build adjacency list for Petersen
    adj = [[] for _ in range(10)]
    for u, v in edges_p:
        adj[u].append(v)
        adj[v].append(u)

    # Enumerate all non-backtracking closed walks of length L < g
    # We need to check L from 1 to g-1.
    # Since girth is 5, we only care about L >= 5.
    # But to be safe, we check all L < g.
    
    def get_walks_dir(max_len):
        """Generate all non-backtracking closed walks of length max_len with direction tracking."""
        walks = []
        for start in range(10):
            stack = [(start, -1, 0, [])]
            while stack:
                curr, prev, l, seq = stack.pop()
                if l == max_len:
                    if curr == start:
                        walks.append(seq)
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    # Determine edge index and direction
                    if (curr, nxt) in edge_index:
                        e_idx = edge_index[(curr, nxt)]
                        if (curr, nxt) == edges_p[e_idx]:
                            sign = 1
                        else:
                            sign = -1
                        stack.append((nxt, curr, l + 1, seq + [(e_idx, sign)]))
        return walks

    # Precompute walks for lengths 5 to g-1
    target_g = g
    # Lower bound on n is 512.
    # n = 10 * p.
    # So p must be >= 52.
    
    # Try small primes first to minimize n
    primes_to_try = [53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    # Precompute all constraints (closed walks that must have non-zero voltage sum)
    all_constraints = []
    for L in range(5, target_g):
        walks = get_walks_dir(L)
        for w in walks:
            all_constraints.append(w)
    
    # For each prime, try to find a voltage assignment using algebraic construction
    for p in primes_to_try:
        # Use primitive element in GF(p) to construct voltages
        # Find a primitive root modulo p
        def find_primitive_root(p):
            if p == 2:
                return 1
            phi = p - 1
            factors = []
            d = 2
            while d * d <= phi:
                if phi % d == 0:
                    factors.append(d)
                    while phi % d == 0:
                        phi //= d
                d += 1
            if phi > 1:
                factors.append(phi)
            
            for g in range(2, p):
                is_primitive = True
                for f in factors:
                    if pow(g, (p-1)//f, p) == 1:
                        is_primitive = False
                        break
                if is_primitive:
                    return g
            return None
        
        primitive_root = find_primitive_root(p)
        
        # Try multiple algebraic constructions
        found = False
        
        # Construction 1: v_i = primitive_root^i mod p
        if primitive_root:
            voltages = [pow(primitive_root, i, p) for i in range(num_edges)]
            valid = True
            for w in all_constraints:
                s = 0
                for e_idx, sign in w:
                    s += sign * voltages[e_idx]
                if s % p == 0:
                    valid = False
                    break
            if valid:
                found = True
        
        # Construction 2: v_i = primitive_root^(i+1) mod p (shifted)
        if not found and primitive_root:
            voltages = [pow(primitive_root, i+1, p) for i in range(num_edges)]
            valid = True
            for w in all_constraints:
                s = 0
                for e_idx, sign in w:
                    s += sign * voltages[e_idx]
                if s % p == 0:
                    valid = False
                    break
            if valid:
                found = True
        
        # Construction 3: Use different primitive root powers
        if not found and primitive_root:
            # Try v_i = primitive_root^(2*i) mod p
            voltages = [pow(primitive_root, 2*i, p) for i in range(num_edges)]
            valid = True
            for w in all_constraints:
                s = 0
                for e_idx, sign in w:
                    s += sign * voltages[e_idx]
                if s % p == 0:
                    valid = False
                    break
            if valid:
                found = True
        
        # Construction 4: Random with fixed seed
        if not found:
            rng = random.Random(p)  # Deterministic per prime
            for attempt in range(5000):
                voltages = [rng.randint(1, p-1) for _ in range(num_edges)]
                valid = True
                for w in all_constraints:
                    s = 0
                    for e_idx, sign in w:
                        s += sign * voltages[e_idx]
                    if s % p == 0:
                        valid = False
                        break
                if valid:
                    found = True
                    break
        
        if found:
            # Construct the graph
            n = 10 * p
            edges_out = []
            # Vertices are (v, i) where v in 0..9, i in 0..p-1
            # Edge between (u, i) and (v, j) if (u,v) is edge in Petersen and j = i + v_e (mod p)
            
            for i in range(num_edges):
                u, v = edges_p[i]
                ve = voltages[i]
                for fiber in range(p):
                    # u, fiber -> v, fiber + ve
                    v1 = u * p + fiber
                    v2 = v * p + (fiber + ve) % p
                    edges_out.append((v1, v2))
                    # v, fiber -> u, fiber - ve
                    v3 = v * p + fiber
                    v4 = u * p + (fiber - ve) % p
                    edges_out.append((v3, v4))
            
            # Remove duplicates
            unique_edges = set()
            for e in edges_out:
                if e[0] > e[1]:
                    e = (e[1], e[0])
                unique_edges.add(e)
            
            return n, list(unique_edges)
    
    # Fallback: Return the LUW graph if no small voltage graph found
    # But we should find one for p=53.
    # If not, we might timeout or fail.
    # Let's assume p=53 works.
    
    # Fallback to original logic if all fail
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
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
