import time
import numpy as np
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    def check_girth(n, edges, g, time_limit):
        if n == 0:
            return -1
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        for i in range(n):
            if len(adj[i]) != k:
                return -1

        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                d = dist[node]
                if d >= g:
                    continue
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = d + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = d + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    def voltage_lift_search():
        if k != 3 or g != 16:
            return None

        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]

        n_base = 10
        edge_index = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i

        adj = [[] for _ in range(n_base)]
        for u, v in pet_edges:
            adj[u].append(v)
            adj[v].append(u)

        max_L = g - 1
        walk_vectors = set()

        for start_v in range(n_base):
            stack = [(start_v, -1, 0, [0] * 15)]
            while stack:
                current, prev, length, counts = stack.pop()

                if length >= max_L:
                    continue

                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = pet_edges[e_idx]
                    d = 1 if current == u else -1

                    new_counts = counts.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1

                    if next_v == start_v and new_l >= 5:
                        vec = tuple(new_counts)
                        if any(x != 0 for x in vec):
                            first_nz = -1
                            for i in range(15):
                                if vec[i] != 0:
                                    first_nz = i
                                    break
                            if first_nz != -1 and vec[first_nz] < 0:
                                vec = tuple(-x for x in vec)
                            walk_vectors.add(vec)

                    stack.append((next_v, current, new_l, new_counts))

        if not walk_vectors:
            return None

        C = np.array(list(walk_vectors), dtype=np.int64)

        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
                i += 6
            return True

        def solve_voltage(m, time_limit):
            v = np.random.randint(1, m, size=15).astype(np.int64)

            def cost(v):
                residues = np.mod(C @ v, m)
                return int(np.sum(residues == 0))

            best_v = v.copy()
            best_cost = cost(v)

            if best_cost == 0:
                return best_v

            T = 1.0
            cooling = 0.999
            start_t = time.time()

            while time.time() - start_t < time_limit:
                if remaining() < 5:
                    break

                j = random.randint(0, 14)
                old_val = v[j]
                v[j] = random.randint(1, m - 1)

                new_cost = cost(v)
                delta = new_cost - best_cost

                if delta < 0 or random.random() < np.exp(-delta / T):
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                        if best_cost == 0:
                            return best_v
                else:
                    v[j] = old_val

                T *= cooling

            return best_v if best_cost == 0 else None

        # Moore bound for k=3, g=16 is 766.
        # n = 10 * m. So 10*m >= 766 => m >= 76.6.
        # Smallest prime >= 77 is 79.

        m = 79
        while m < 300:
            if remaining() < 10:
                break

            if is_prime(m):
                t_limit = min(5.0, remaining() / 5.0)

                v = solve_voltage(m, t_limit)

                if v is not None:
                    n = 10 * m
                    edges = []
                    for u, v_base in pet_edges:
                        e_idx = edge_index[(u, v_base)]
                        vol = v[e_idx]
                        for layer in range(m):
                            u_lift = u * m + layer
                            v_lift = v_base * m + (layer + vol) % m
                            if u_lift < v_lift:
                                edges.append((u_lift, v_lift))
                            else:
                                edges.append((v_lift, u_lift))

                    gc = check_girth(n, edges, g, deadline - 2)
                    if gc >= g:
                        return (n, edges)

            m += 1

        return None

    def cayley_graph_search():
        if k != 3 or g != 16:
            return None

        # Use a group that is a product of small cyclic groups
        # We want to find generators S = {a, b, c} such that the Cayley graph
        # has girth >= 16.

        # For simplicity, try Z_p x Z_q for small primes p, q
        # The group has p*q elements.

        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
                i += 6
            return True

        # Try different group sizes
        # Moore bound is 766, so we need p*q >= 766
        # But we want to find the smallest, so try from smaller

        # Actually, for Cayley graphs, the structure is different.
        # Let's try a few small products

        # For now, try Z_p for prime p >= 767
        # The smallest prime >= 767 is 769

        # But Z_p with 3 generators might be hard to get girth 16
        # Let's try a different approach

        # Use a non-abelian group: S_5 has 120 elements, too small
        # A_5 has 60, too small

        # Let's try Z_p x Z_q where p, q are primes
        # p*q >= 766
        # Small pairs: (2, 383), (3, 257), (5, 157), (7, 109), (11, 73), (13, 61), (17, 47), (19, 43), (23, 37), (29, 31)

        # For each pair, try random generators

        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
            if remaining() < 10:
                break
            q = 766 // p + 1
            # Find smallest prime >= q
            while not is_prime(q):
                q += 1

            n = p * q

            # Group elements are (a, b) with a in Z_p, b in Z_q
            # Represent as a * q + b

            # Try random generators
            for _ in range(100):
                if remaining() < 5:
                    break

                # Random generators
                a1 = random.randint(1, p - 1) if p > 1 else 0
                b1 = random.randint(1, q - 1)
                a2 = random.randint(1, p - 1) if p > 1 else 0
                b2 = random.randint(1, q - 1)
                a3 = random.randint(1, p - 1) if p > 1 else 0
                b3 = random.randint(1, q - 1)

                # Make sure they are distinct and not identity
                gens = [(a1, b1), (a2, b2), (a3, b3)]

                # Check they are distinct
                if len(set(gens)) != 3:
                    continue

                # Build Cayley graph
                edges = []
                for a in range(p):
                    for b in range(q):
                        u = a * q + b
                        for ga, gb in gens:
                            va = (a + ga) % p
                            vb = (b + gb) % q
                            v = va * q + vb
                            if u < v:
                                edges.append((u, v))
                            else:
                                edges.append((v, u))

                # Check for duplicate edges
                edge_set = set()
                valid = True
                for e in edges:
                    if e in edge_set:
                        valid = False
                        break
                    edge_set.add(e)

                if not valid:
                    continue

                # Check girth
                gc = check_girth(n, edges, g, deadline - 2)
                if gc >= g:
                    return (n, edges)

        return None

    # Try voltage lift first (more likely to work)
    res = voltage_lift_search()
    if res is not None:
        return res

    # If voltage lift fails, try Cayley graph
    res = cayley_graph_search()
    if res is not None:
        return res

    return (0, [])

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
