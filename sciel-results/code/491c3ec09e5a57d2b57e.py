import random
import numpy as np

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

    def get_walks_dir(max_len):
        """Generate all non-backtracking closed walks of length max_len with direction info."""
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
                    e_idx = edge_index[(curr, nxt)]
                    if (curr, nxt) == edges_p[e_idx]:
                        sign = 1
                    else:
                        sign = -1
                    stack.append((nxt, curr, l + 1, seq + [(e_idx, sign)]))
        return walks

    target_g = g
    # Lower bound on n is 512.
    # n = 10 * p.
    # So p must be >= 52.
    
    primes_to_try = [53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149]
    
    # Precompute all non-backtracking closed walks for lengths 5 to g-1
    all_constraints = []
    for L in range(5, target_g):
        walks = get_walks_dir(L)
        for w in walks:
            all_constraints.append(w)
    
    def is_valid_voltage(voltages, p):
        """Check if voltage assignment satisfies all constraints."""
        for w in all_constraints:
            s = 0
            for e_idx, sign in w:
                s += sign * voltages[e_idx]
            if s % p == 0:
                return False
        return True

    def try_structured_voltage(p):
        """Try structured voltage assignments based on primitive elements."""
        # Try voltage assignment: v_i = a^i for some generator a in GF(p)
        # First find a primitive root modulo p
        def is_primitive_root(a, p):
            if a <= 1 or a >= p:
                return False
            # Check that a^((p-1)/q) != 1 for all prime factors q of p-1
            n = p - 1
            # Find prime factors of n
            factors = set()
            temp = n
            d = 2
            while d * d <= temp:
                while temp % d == 0:
                    factors.add(d)
                    temp //= d
                d += 1
            if temp > 1:
                factors.add(temp)
            
            for q in factors:
                if pow(a, (p-1)//q, p) == 1:
                    return False
            return True
        
        # Find primitive root
        primitive_root = None
        for a in range(2, p):
            if is_primitive_root(a, p):
                primitive_root = a
                break
        
        if primitive_root is None:
            return None
        
        # Try different structured assignments
        # Pattern 1: v_i = primitive_root^i
        # Pattern 2: v_i = i * primitive_root
        # Pattern 3: v_i = primitive_root^(i * 2)
        # Pattern 4: Mix of patterns
        
        patterns = []
        
        # Pattern 1: Powers of primitive root
        patterns.append([pow(primitive_root, i, p) for i in range(15)])
        
        # Pattern 2: Linear times primitive root
        patterns.append([(i + 1) * primitive_root % p for i in range(15)])
        
        # Pattern 3: Every other power
        patterns.append([pow(primitive_root, i * 2, p) for i in range(15)])
        
        # Pattern 4: Alternating
        patterns.append([pow(primitive_root, i if i % 2 == 0 else i + 7, p) for i in range(15)])
        
        # Pattern 5: Fibonacci-like in exponent
        fib = [1, 1]
        for i in range(13):
            fib.append((fib[-1] + fib[-2]) % (p - 1))
        patterns.append([pow(primitive_root, fib[i], p) for i in range(15)])
        
        # Pattern 6: Quadratic
        patterns.append([(i * i + i + 1) % p for i in range(15)])
        
        # Pattern 7: Cubic
        patterns.append([(i * i * i + i * i + i + 1) % p for i in range(15)])
        
        # Try each pattern
        for pattern in patterns:
            if all(v != 0 for v in pattern):  # Avoid zero voltages
                if is_valid_voltage(pattern, p):
                    return pattern
        
        return None

    def try_random_voltage(p, num_attempts=5000):
        """Try random voltage assignments."""
        rng = random.Random(42)
        for attempt in range(num_attempts):
            voltages = [rng.randint(1, p-1) for _ in range(15)]
            if is_valid_voltage(voltages, p):
                return voltages
        return None

    # Try to find a valid voltage assignment
    for p in primes_to_try:
        # First try structured assignments
        voltages = try_structured_voltage(p)
        
        # If structured fails, try random
        if voltages is None:
            voltages = try_random_voltage(p)
        
        if voltages is not None:
            # Construct the graph
            n = 10 * p
            edges_out = set()
            
            for i in range(num_edges):
                u, v = edges_p[i]
                ve = voltages[i]
                for fiber in range(p):
                    # u, fiber -> v, fiber + ve
                    v1 = u * p + fiber
                    v2 = v * p + (fiber + ve) % p
                    if v1 < v2:
                        edges_out.add((v1, v2))
                    else:
                        edges_out.add((v2, v1))
                    # v, fiber -> u, fiber - ve
                    v3 = v * p + fiber
                    v4 = u * p + (fiber - ve) % p
                    if v3 < v4:
                        edges_out.add((v3, v4))
                    else:
                        edges_out.add((v4, v3))
            
            return n, list(edges_out)
    
    # Fallback: Return a larger graph if no small voltage graph found
    # Use p = 151 or larger
    p = 151
    rng = random.Random(123)
    for attempt in range(10000):
        voltages = [rng.randint(1, p-1) for _ in range(15)]
        if is_valid_voltage(voltages, p):
            break
    
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
            v3 = v * p + fiber
            v4 = u * p + (fiber - ve) % p
            if v3 < v4:
                edges_out.add((v3, v4))
            else:
                edges_out.add((v4, v3))
    
    return n, list(edges_out)

