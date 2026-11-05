import random
import time
import math

def construct():
    deadline = time.time() + 115  # Safety margin
    n = 40

    # Bitmask utilities for fast adjacency
    FULL = (1 << n) - 1

    def make_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            bits = adj[i]
            while bits:
                lsb = bits & -bits
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                bits ^= lsb
        return edges

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def creates_c6(adj, u, v):
        """
        Check if adding edge (u, v) creates a C6.
        Equivalent to: is there a path of length 5 from u to v?
        Path: u - a - b - c - d - v
        We check if there exists a in N(u), d in N(v) such that
        there is a path a - b - c - d (length 3).
        """
        nu = adj[u] & ~(1 << v)  # neighbors of u excluding v
        nv = adj[v] & ~(1 << u)  # neighbors of v excluding u

        if nu == 0 or nv == 0:
            return False

        # For each a in N(u), d in N(v), check if a and d are connected by path of length 3
        # a - b - c - d
        # This is: exists b in N(a), c in N(b) such that c in N(d)
        # i.e., N(a) & N(N(d)) != 0 but with distinctness constraints

        # More carefully:
        # For a in N(u), d in N(v), a != d:
        #   Check if there exists b in N(a) \ {u, v, a, d} and c in N(b) \ {u, v, a, b, d}
        #   such that c in N(d).
        #   Equivalently: exists b in N(a) \ {u, v, a, d} such that
        #   (N(b) & N(d)) & ~(1<<u) & ~(1<<v) & ~(1<<a) & ~(1<<b) & ~(1<<d) != 0

        bits_a = nu
        while bits_a:
            lsb_a = bits_a & -bits_a
            a = lsb_a.bit_length() - 1
            bits_a ^= lsb_a

            # d must be in N(v) and d != a, d != u, d != v
            # d in nv, d != a
            bits_d = nv & ~(1 << a)
            if bits_d == 0:
                continue

            # For this a, check all d in bits_d
            # b in N(a) \ {u, v, a, d}
            # We can iterate d and for each, check if there's a valid b

            bits_d_cur = bits_d
            while bits_d_cur:
                lsb_d = bits_d_cur & -bits_d_cur
                d = lsb_d.bit_length() - 1
                bits_d_cur ^= lsb_d

                # b in N(a) excluding u, v, a, d
                b_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if b_mask == 0:
                    continue

                # For each b in b_mask, check if N(b) & N(d) has any vertex c
                # c != u, v, a, b, d
                c_mask = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if c_mask == 0:
                    continue

                bits_b = b_mask
                while bits_b:
                    lsb_b = bits_b & -bits_b
                    b = lsb_b.bit_length() - 1
                    bits_b ^= lsb_b

                    # c in N(b) & c_mask, c != b
                    if adj[b] & c_mask & ~(1 << b):
                        return True

        return False

    def is_c6_free(adj):
        """Full validation: check that no C6 exists in the graph."""
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    # Temporarily remove edge and check if path of length 5 exists
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    if creates_c6(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        return False
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return True

    def get_missing_edges(adj):
        missing = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    missing.append((u, v))
        return missing

    def get_existing_edges(adj):
        existing = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    existing.append((u, v))
        return existing

    # Greedy construction
    def greedy_construct(rng):
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)

        for u, v in all_edges:
            if not creates_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    rng = random.Random(42)

    # Start with random restarts of greedy
    best_adj = None
    best_m = 0
    initial_time = time.time()

    while time.time() < initial_time + 8:
        adj = greedy_construct(rng)
        m = count_edges(adj)
        if m > best_m:
            best_m = m
            best_adj = adj[:]

    current_adj = best_adj[:]
    current_m = best_m

    # Simulated Annealing with bitmask
    T = 1.0
    time_start = time.time()
    iter_count = 0
    last_cool = time.time()

    while time.time() < deadline:
        iter_count += 1
        if iter_count % 500 == 0:
            T *= 0.98

        move_type = rng.random()

        valid = False
        delta_m = 0

        if move_type < 0.5:
            # Add an edge
            missing = get_missing_edges(current_adj)
            if missing:
                u, v = rng.choice(missing)
                if not creates_c6(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    delta_m = 1
                    valid = True
        elif move_type < 0.7:
            # Remove an edge
            existing = get_existing_edges(current_adj)
            if existing:
                u, v = rng.choice(existing)
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                delta_m = -1
                valid = True
        else:
            # 1-Swap: remove one, add one
            existing = get_existing_edges(current_adj)
            missing = get_missing_edges(current_adj)
            if existing and missing:
                u1, v1 = rng.choice(existing)
                u2, v2 = rng.choice(missing)

                # Temporarily remove (u1, v1)
                current_adj[u1] &= ~(1 << v1)
                current_adj[v1] &= ~(1 << u1)

                if not creates_c6(current_adj, u2, v2):
                    current_adj[u2] |= (1 << v2)
                    current_adj[v2] |= (1 << u2)
                    delta_m = 0
                    valid = True
                else:
                    # Revert
                    current_adj[u1] |= (1 << v1)
                    current_adj[v1] |= (1 << u1)
                    valid = False

        if valid:
            if delta_m >= 0 or (T > 0.001 and rng.random() < math.exp(delta_m / T)):
                current_m += delta_m
                if current_m > best_m:
                    best_m = current_m
                    best_adj = current_adj[:]
            else:
                # Reject and revert
                if move_type < 0.5:
                    # Add was applied, revert
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                elif move_type < 0.7:
                    # Remove was applied, revert
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                else:
                    # Swap: remove (u2,v2), add (u1,v1)
                    current_adj[u2] &= ~(1 << v2)
                    current_adj[v2] &= ~(1 << u2)
                    current_adj[u1] |= (1 << v1)
                    current_adj[v1] |= (1 << u1)

    # Final validation
    if not is_c6_free(best_adj):
        # Fallback: rebuild from scratch with greedy
        adj = greedy_construct(rng)
        if is_c6_free(adj):
            best_adj = adj

    return adj_to_edges(best_adj)
