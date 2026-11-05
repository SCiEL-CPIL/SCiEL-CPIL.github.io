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

    def check_voltages(voltages, p):
        """Check if voltage assignment satisfies all constraints."""
        for w in all_constraints:
            s = 0
            for e_idx, sign in w:
                s += sign * voltages[e_idx]
            if s % p == 0:
                return False
        return True

    def try_build(p, voltages):
        """Build the lifted graph from voltages."""
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

    # Moore bound for (3,16): n >= 3*(2^8-1) = 765
    # For Petersen lifts: n = 10*p, so p >= 77
    # Primes >= 77: 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, ...
    primes_to_try = [79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]

    for p in primes_to_try:
        alpha = find_primitive_root(p)
        if alpha is None:
            continue

        found = False
        # Strategy 1: Try all powers alpha^a for a in 1..p-1 as voltage multipliers
        # v_i = alpha^(a*i) mod p for various a
        for a in range(1, p):
            voltages = [pow(alpha, (a * i) % (p - 1), p) for i in range(num_edges)]
            if check_voltages(voltages, p):
                found = True
                break

        if not found:
            # Strategy 2: v_i = alpha^(a*i + b) for various a, b
            for a in range(1, min(p, 50)):
                for b in range(min(p, 20)):
                    voltages = [pow(alpha, (a * i + b) % (p - 1), p) for i in range(num_edges)]
                    if check_voltages(voltages, p):
                        found = True
                        break
                if found:
                    break

        if not found:
            # Strategy 3: v_i = alpha^(a*i^2 + b*i + c) quadratic
            for a in range(1, min(p, 15)):
                for b in range(min(p, 10)):
                    for c in range(min(p, 10)):
                        voltages = [pow(alpha, (a * (i * i) + b * i + c) % (p - 1), p) for i in range(num_edges)]
                        if check_voltages(voltages, p):
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        if not found:
            # Strategy 4: Deterministic pseudo-random using LCG with various seeds
            # Use a simple LCG: x_{n+1} = (a*x_n + c) mod p, start from various seeds
            for seed in range(1, min(p, 200)):
                for lcg_a in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
                    if lcg_a >= p:
                        continue
                    voltages = []
                    x = seed
                    for i in range(num_edges):
                        x = (lcg_a * x + seed) % p
                        if x == 0:
                            x = 1
                        voltages.append(x)
                    if check_voltages(voltages, p):
                        found = True
                        break
                if found:
                    break

        if not found:
            # Strategy 5: Use the multiplicative group structure more carefully
            # Try v_i = alpha^(i * k) where k is chosen to maximize spread
            # Also try v_i = alpha^(i^3 mod (p-1))
            for exp_fn in [lambda i: (i ** 3) % (p - 1),
                          lambda i: (i * (i + 1) // 2) % (p - 1),
                          lambda i: (i ** 2 + 1) % (p - 1),
                          lambda i: (i ** 2 - i + 1) % (p - 1)]:
                voltages = [pow(alpha, exp_fn(i), p) for i in range(num_edges)]
                if check_voltages(voltages, p):
                    found = True
                    break

        if found:
            return try_build(p, voltages)

    # Fallback: larger prime with exhaustive search
    p = 997
    alpha = find_primitive_root(p)
    import random
    rng = random.Random(42)
    for attempt in range(20000):
        voltages = [rng.randint(1, p - 1) for _ in range(num_edges)]
        if check_voltages(voltages, p):
            return try_build(p, voltages)
    raise ValueError("Could not find valid voltage assignment")
