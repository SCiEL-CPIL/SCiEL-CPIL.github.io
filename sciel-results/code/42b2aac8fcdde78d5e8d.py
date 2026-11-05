import time
import numpy as np
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

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
        C = np.unique(C, axis=0)
        n_constraints = C.shape[0]

        def solve_voltage(m, time_limit):
            best_v = None
            best_cost = float('inf')
            start_t = time.time()

            # Precompute: for each column j, C[:, j]
            # cost(v) = sum over rows i of [C[i] @ v == 0 mod m]
            
            def compute_cost(v):
                residues = np.mod(C @ v, m)
                return int(np.sum(residues == 0))

            max_restarts = max(1, int(time_limit / 0.3))
            for restart in range(max_restarts):
                if time.time() - start_t > time_limit:
                    break
                if remaining() < 5:
                    break

                v = np.random.randint(1, m, size=15).astype(np.int64)
                cur_cost = compute_cost(v)
                if cur_cost == 0:
                    return v.copy()
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_v = v.copy()

                # Simulated annealing with better schedule
                T = max(1.0, cur_cost * 0.5)
                cooling = 0.9998
                min_T = 0.05
                stuck = 0
                best_this_restart = cur_cost

                while time.time() - start_t < time_limit and remaining() > 5:
                    j = random.randint(0, 14)
                    old_val = v[j]
                    new_val = random.randint(1, m - 1)
                    if new_val == old_val:
                        continue
                    v[j] = new_val
                    new_cost = compute_cost(v)
                    delta = new_cost - cur_cost

                    if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                        cur_cost = new_cost
                        if new_cost < best_this_restart:
                            best_this_restart = new_cost
                            stuck = 0
                        else:
                            stuck += 1
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_v = v.copy()
                        if best_cost == 0:
                            return best_v
                    else:
                        v[j] = old_val
                        stuck += 1

                    T = max(min_T, T * cooling)

                    # If stuck too long, restart this SA run
                    if stuck > 200:
                        break

            return best_v if best_cost == 0 else None

        # Try all m values starting from 77 (Moore bound: n >= 766, m >= 77)
        m = 77
        while m < 200:
            if remaining() < 15:
                break
            t_limit = min(10.0, max(2.0, remaining() / 10.0))
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
                gc = check_girth(n, edges, g, deadline - 3)
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
