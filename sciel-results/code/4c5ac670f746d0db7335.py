import time
import numpy as np
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification ----------------
    def check_girth(n, edges, g, time_limit):
        if n == 0:
            return -1
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        # Check degree
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
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- Voltage Lift on Petersen Graph ----------------
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

        # Generate constraints
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
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True

        def solve_voltage(m, time_limit):
            # Local repair approach
            max_attempts = 500
            for attempt in range(max_attempts):
                if time.time() > time_limit:
                    break

                # Random initialization
                x = np.random.randint(1, m, size=15).astype(np.int64)

                # Local Repair Loop
                for repair_step in range(200):
                    if time.time() > time_limit:
                        break

                    # Check constraints
                    S = (C @ x) % m
                    violated_indices = np.where(S == 0)[0]

                    if len(violated_indices) == 0:
                        return x

                    # Pick a random violated constraint
                    v_idx = violated_indices[random.randint(0, len(violated_indices) - 1)]
                    violated_c = C[v_idx]

                    # Pick a random edge with non-zero coefficient in this constraint
                    nonzero_indices = np.where(violated_c != 0)[0]
                    if len(nonzero_indices) == 0:
                        break
                    j = nonzero_indices[random.randint(0, len(nonzero_indices) - 1)]

                    # Try to change x[j] to fix this constraint
                    # Current sum is 0.
                    # We want new_sum = (0 + c_j * (new_val - old_val)) % m != 0
                    # c_j * (new_val - old_val) != 0 mod m
                    # Since c_j != 0 and m is prime, new_val != old_val mod m is sufficient?
                    # Not quite, we need the product to be non-zero.
                    # If c_j is invertible (it is, m prime, c_j not 0 mod m? c_j can be > m? No, c_j is small integer.
                    # If c_j % m == 0, then changing x_j doesn't help.
                    # But c_j is small (max walk length 15), m >= 77. So c_j != 0 mod m.

                    old_val = x[j]
                    fixed = False
                    for _ in range(10):
                        new_val = random.randint(1, m - 1)
                        if new_val == old_val:
                            continue
                        # Check if this fixes the current constraint
                        # new_sum = (S[v_idx] + c_j * (new_val - old_val)) % m
                        # S[v_idx] is 0.
                        # So new_sum = (c_j * (new_val - old_val)) % m
                        if (violated_c[j] * (new_val - old_val)) % m != 0:
                            x[j] = new_val
                            fixed = True
                            break

                    if not fixed:
                        # If we can't fix it easily, just randomize x[j]
                        x[j] = random.randint(1, m - 1)

            return None

        # Moore bound for k=3, g=16 is 766.
        # n = 10 * m. So 10*m >= 766 => m >= 76.6.
        # Smallest prime >= 77 is 79.

        # Iterate primes in increasing order to find smallest n.
        m = 79
        while m < 300:
            if remaining() < 10:
                break

            if is_prime(m):
                # Allocate more time to smaller m?
                # Or just fixed time.
                t_limit = min(5.0, remaining() / 5.0)

                v = solve_voltage(m, t_limit)

                if v is not None:
                    # Construct graph
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

                    # Verify
                    gc = check_girth(n, edges, g, deadline - 2)
                    if gc >= g:
                        return (n, edges)

            m += 1

        return None

    res = voltage_lift_search()
    if res is not None:
        return res

    return (0, [])

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
