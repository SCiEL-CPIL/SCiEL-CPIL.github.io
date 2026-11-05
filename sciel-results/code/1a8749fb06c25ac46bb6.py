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

    # Convert constraints to a NumPy matrix for fast vectorized checking
    # M[j, i] is the coefficient of edge i in constraint j
    num_constraints = len(all_constraints)
    M = np.zeros((num_constraints, num_edges), dtype=np.int64)
    for j, w in enumerate(all_constraints):
        for e_idx, sign in w:
            M[j, e_idx] += sign

    def check_voltages_fast(voltages, p):
        """Check if voltage assignment satisfies all constraints using NumPy."""
        v = np.array(voltages, dtype=np.int64)
        # Compute (M @ v) % p
        # Note: M has small integer entries, v has entries < p
        # We can do the dot product in integers then mod p
        sums = M.dot(v) % p
        # If any sum is 0, a constraint is violated
        return not np.any(sums == 0)

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
            ve = voltages[i] % p
            for fiber in range(p):
                v1 = u * p + fiber
                v2 = v * p + (fiber + ve) % p
                if v1 < v2:
                    edges_out.add((v1, v2))
                else:
                    edges_out.add((v2, v1))
        return n, list(edges_out)

    # Moore bound for (3,16): n >= 765. Since n=10p, p >= 77.
    # We try primes starting from 79.
    # We need to find a voltage assignment for the smallest p.
    
    # Generate list of primes
    def get_primes(start, count):
        primes = []
        cand = start
        while len(primes) < count:
            is_prime = True
            if cand < 2:
                is_prime = False
            else:
                for d in range(2, int(cand**0.5) + 1):
                    if cand % d == 0:
                        is_prime = False
                        break
            if is_prime:
                primes.append(cand)
            cand += 1
        return primes

    primes_to_try = get_primes(79, 50)

    for p in primes_to_try:
        alpha = find_primitive_root(p)
        if alpha is None:
            continue

        found = False
        best_voltages = None

        # Strategy 1: v_i = alpha^(a*i + b) mod p
        # This generates multiplicative sequences.
        # Note: The check is additive: sum(v_e * coeff) == 0.
        # If v_e are powers of alpha, this is a linear recurrence check.
        
        # Let's try a broader set of structured vectors.
        # 1. Linear: v_i = (a*i + b) % p
        for a in range(1, min(50, p)):
            for b in range(0, min(50, p)):
                voltages = [(a*i + b) % p for i in range(num_edges)]
                if check_voltages_fast(voltages, p):
                    found = True
                    best_voltages = voltages
                    break
            if found:
                break

        if not found:
            # 2. Exponential: v_i = alpha^(a*i + b) mod p
            if alpha:
                for a in range(1, min(50, p-1)):
                    for b in range(0, min(50, p-1)):
                        voltages = [pow(alpha, (a*i + b) % (p-1), p) for i in range(num_edges)]
                        if check_voltages_fast(voltages, p):
                            found = True
                            best_voltages = voltages
                            break
                    if found:
                        break

        if not found:
            # 3. Quadratic: v_i = (a*i^2 + b*i + c) % p
            for a in range(1, min(30, p)):
                for b in range(0, min(30, p)):
                    for c in range(0, min(30, p)):
                        voltages = [(a*i*i + b*i + c) % p for i in range(num_edges)]
                        if check_voltages_fast(voltages, p):
                            found = True
                            best_voltages = voltages
                            break
                    if found:
                        break
                if found:
                    break

        if not found:
            # 4. Randomized search with NumPy acceleration
            # Generate many random vectors and check in batch?
            # Since check_voltages_fast uses dot product, we can batch multiple vectors.
            # Let's create a batch of 1000 random vectors.
            num_trials = 1000
            for trial in range(num_trials):
                # Deterministic pseudo-random
                state = 12345 + trial * 7919
                voltages = []
                for i in range(num_edges):
                    state = (state * 1103515245 + 12345) % (2**31)
                    v = (state % (p-1)) + 1 # Non-zero? Additive group includes 0.
                    # Actually, voltages can be 0.
                    v = state % p
                    voltages.append(v)
                if check_voltages_fast(voltages, p):
                    found = True
                    best_voltages = voltages
                    break

        if found:
            return build_graph(best_voltages, p)

    # Fallback: larger prime
    p = 257
    alpha = find_primitive_root(p)
    for trial in range(10000):
        state = 12345 + trial * 104729
        voltages = []
        for i in range(num_edges):
            state = (state * 1103515245 + 12345) % (2**31)
            v = state % p
            voltages.append(v)
        if check_voltages_fast(voltages, p):
            return build_graph(voltages, p)

    raise ValueError("Could not find valid voltage assignment")
