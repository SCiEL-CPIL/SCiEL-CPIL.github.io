import numpy as np
from itertools import product

def construct(k, g):
    # Petersen graph: 10 vertices, 15 edges, girth 5
    edges_p = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    num_edges = 15
    edge_index = {}
    for i, e in enumerate(edges_p):
        edge_index[e] = i
        edge_index[(e[1], e[0])] = i

    adj = [[] for _ in range(10)]
    for u, v in edges_p:
        adj[u].append(v)
        adj[v].append(u)

    # Generate all non-backtracking closed walks of length L in Petersen graph
    def get_walks_dir(max_len):
        walks = []
        for start in range(10):
            stack = [(start, -1, 0, [])]
            while stack:
                curr, prev, l, seq = stack.pop()
                if l == max_len:
                    if curr == start:
                        walks.append(tuple(seq))
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    e_idx = edge_index[(curr, nxt)]
                    if (curr, nxt) == edges_p[e_idx]:
                        sign = 1
                    else:
                        sign = -1
                    stack.append((nxt, curr, l + 1, seq + [(e_idx, sign)]))
        return walks

    # Precompute all constraint walks for lengths 5 to g-1
    target_g = g
    all_constraints = []
    for L in range(5, target_g):
        walks = get_walks_dir(L)
        all_constraints.extend(walks)

    # Convert constraints to coefficient vectors for efficient checking
    # Each constraint is a vector in Z_p^15
    # Represent as list of (edge_idx, sign) pairs
    # For speed, precompute as numpy arrays
    constraint_matrix = []
    for w in all_constraints:
        vec = [0] * num_edges
        for e_idx, sign in w:
            vec[e_idx] += sign
        constraint_matrix.append(vec)
    
    constraint_matrix_np = np.array(constraint_matrix, dtype=np.int64)

    def find_primitive_root(p):
        """Find a primitive root modulo prime p."""
        if p == 2:
            return 1
        n = p - 1
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.append(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.append(n)
        for g in range(2, p):
            ok = True
            for f in factors:
                if pow(g, (p-1)//f, p) == 1:
                    ok = False
                    break
            if ok:
                return g
        return None

    def check_voltages_fast(voltages_np, p):
        """Check if voltage assignment satisfies all constraints using numpy."""
        # For each constraint, compute sum(sign * voltage) mod p
        # constraint_matrix_np has shape (num_constraints, num_edges)
        # voltages_np has shape (num_edges,)
        # Compute dot products
        sums = constraint_matrix_np @ voltages_np
        # Check if any sum is 0 mod p
        return not np.any(sums % p == 0)

    # List of primes to try (must have 10*p >= 765, so p >= 77)
    primes_to_try = [79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]

    for p in primes_to_try:
        alpha = find_primitive_root(p)
        if alpha is None:
            continue
        
        # Precompute powers of alpha
        alpha_powers = [pow(alpha, i, p) for i in range(p-1)]
        
        found = False
        
        # Strategy: try voltage assignments of form v_i = alpha^(a*i + b) mod p
        # and also structured assignments based on graph edges
        # Outer cycle: edges 0-4, Inner star: edges 5-9, Spokes: edges 10-14
        
        # Try various linear exponent patterns
        for a in range(1, min(p-1, 50)):
            for b in range(0, min(p-1, 20)):
                voltages = [alpha_powers[(a*i + b) % (p-1)] for i in range(num_edges)]
                voltages_np = np.array(voltages, dtype=np.int64)
                if check_voltages_fast(voltages_np, p):
                    found = True
                    break
            if found:
                break
        
        if not found:
            # Try quadratic exponent patterns: v_i = alpha^(a*i^2 + b*i + c)
            for a in range(1, min(p-1, 10)):
                for b in range(0, min(p-1, 10)):
                    for c in range(0, min(p-1, 5)):
                        voltages = [alpha_powers[(a*(i*i) + b*i + c) % (p-1)] for i in range(num_edges)]
                        voltages_np = np.array(voltages, dtype=np.int64)
                        if check_voltages_fast(voltages_np, p):
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        if not found:
            # Try structured assignments: different patterns for outer, inner, spokes
            # Outer edges (0-4): alpha^(a*i + b)
            # Inner edges (5-9): alpha^(c*(i-5) + d)
            # Spoke edges (10-14): alpha^(e*(i-10) + f)
            for a in range(1, min(p-1, 10)):
                for b in range(0, min(p-1, 5)):
                    for c in range(1, min(p-1, 10)):
                        for d in range(0, min(p-1, 5)):
                            for e in range(1, min(p-1, 10)):
                                for f in range(0, min(p-1, 5)):
                                    voltages = []
                                    for i in range(5):
                                        voltages.append(alpha_powers[(a*i + b) % (p-1)])
                                    for i in range(5):
                                        voltages.append(alpha_powers[(c*i + d) % (p-1)])
                                    for i in range(5):
                                        voltages.append(alpha_powers[(e*i + f) % (p-1)])
                                    voltages_np = np.array(voltages, dtype=np.int64)
                                    if check_voltages_fast(voltages_np, p):
                                        found = True
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
        
        if found:
            # Construct the lifted graph
            n = 10 * p
            edges_out = set()
            for i in range(num_edges):
                u, v = edges_p[i]
                ve = voltages[i]
                for fiber in range(p):
                    v1 = u * p + fiber
                    v2 = v * p + (fiber + ve) % p
                    if v1 < v2:
                        edges_out.add((v1, v2))
                    else:
                        edges_out.add((v2, v1))
            return n, list(edges_out)

    # Fallback: use random search with more attempts
    import random
    rng = random.Random(42)
    for p in primes_to_try:
        alpha = find_primitive_root(p)
        if alpha is None:
            continue
        alpha_powers = [pow(alpha, i, p) for i in range(p-1)]
        
        for attempt in range(20000):
            # Random voltage assignment using powers of primitive root
            exponents = [rng.randint(0, p-2) for _ in range(num_edges)]
            voltages = [alpha_powers[e] for e in exponents]
            voltages_np = np.array(voltages, dtype=np.int64)
            if check_voltages_fast(voltages_np, p):
                n = 10 * p
                edges_out = set()
                for i in range(num_edges):
                    u, v = edges_p[i]
                    ve = voltages[i]
                    for fiber in range(p):
                        v1 = u * p + fiber
                        v2 = v * p + (fiber + ve) % p
                        if v1 < v2:
                            edges_out.add((v1, v2))
                        else:
                            edges_out.add((v2, v1))
                return n, list(edges_out)
    
    raise ValueError("Could not find valid voltage assignment")
