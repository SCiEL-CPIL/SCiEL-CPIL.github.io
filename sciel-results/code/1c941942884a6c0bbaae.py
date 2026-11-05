import numpy as np

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

    # Convert constraints to sparse coefficient vectors for fast checking
    # Each constraint: list of (edge_idx, sign) -> convert to list of (edge_idx, coeff)
    constraint_sparse = []
    for w in all_constraints:
        coeff_dict = {}
        for e_idx, sign in w:
            coeff_dict[e_idx] = coeff_dict.get(e_idx, 0) + sign
        # Remove zero coefficients
        sparse = [(idx, c) for idx, c in coeff_dict.items() if c != 0]
        if sparse: # If a walk visits an edge in both directions, they cancel out. 
                   # If the walk is a back-and-forth, it would be length 2, which is < 5, so not in list.
                   # But non-backtracking walks can still have cancelling edges if they traverse a cycle?
                   # No, in a simple graph, a non-backtracking walk that traverses an edge and then later traverses it back
                   # implies a cycle. The voltage sum is the sum of signs.
                   # We just need to check if the sum is 0 mod p.
                   # So we keep the sparse representation.
        constraint_sparse.append(sparse)

    def check_voltages_fast(voltages, p):
        """Check if voltage assignment satisfies all constraints using sparse dot products."""
        for sparse in constraint_sparse:
            s = 0
            for idx, c in sparse:
                s += c * voltages[idx]
            if s % p == 0:
                return False
        return True

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

    def build_graph(voltages, p):
        """Construct the lifted graph from voltage assignment."""
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

    # We want to minimize n = 10 * p.
    # Try primes starting from the smallest possible that could work.
    # Moore bound for (3,16) is 765. So p >= 77.
    # Primes: 79, 83, 89, 97...
    
    primes_to_try = [79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    
    for p in primes_to_try:
        # Use a deterministic PRNG to generate random voltage assignments
        # We use a simple LCG for speed and determinism
        # We want voltages in 1..p-1 (non-zero)
        # Actually, voltages can be 0? If voltage is 0, the lift is a disjoint union of copies.
        # But we need a connected graph? The task says "construct a 3-regular graph". It doesn't explicitly say connected,
        # but "cage graph" usually implies connected. The voltage lift is connected if the voltage assignment generates the whole group Z_p.
        # If any voltage is 0, the graph might be disconnected. 
        # However, the girth condition is local. 
        # But standard cage graphs are connected. 
        # Let's ensure voltages are non-zero to be safe, or check connectivity if needed.
        # Actually, if the graph is disconnected, the girth is the min girth of components.
        # If we use random non-zero voltages, it's likely connected if p is prime and the generated subgroup is Z_p.
        
        found = False
        best_voltages = None
        
        # Try many random assignments
        # We use a seed based on p and trial number to be deterministic
        for trial in range(50000):
            # Simple fast PRNG
            state = (trial * 1103515245 + 12345 + p) % (2**31)
            voltages = []
            for i in range(num_edges):
                state = (state * 1103515245 + 12345) % (2**31)
                v = (state % (p - 1)) + 1 # 1 to p-1
                voltages.append(v)
            
            if check_voltages_fast(voltages, p):
                found = True
                best_voltages = voltages
                break
        
        if found:
            return build_graph(best_voltages, p)

    # If we get here, we haven't found a solution for small primes.
    # This is unlikely for p=79,83,89 given the number of constraints.
    # We can fall back to a larger prime if necessary, but the score will be worse.
    # For now, raise an error or return a larger one.
    # Let's try p=127 just in case.
    p = 127
    for trial in range(50000):
        state = (trial * 1103515245 + 12345 + p) % (2**31)
        voltages = []
        for i in range(num_edges):
            state = (state * 1103515245 + 12345) % (2**31)
            v = (state % (p - 1)) + 1
            voltages.append(v)
        
        if check_voltages_fast(voltages, p):
            return build_graph(voltages, p)

    raise ValueError("Could not find valid voltage assignment")
