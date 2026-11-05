import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification (BFS-based, fast) ----------------
    def check_girth(n, edges, min_girth, time_limit):
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
                        if cycle_len < min_girth:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < min_girth:
                return min_cycle
        return min_cycle

    # ---------------- Voltage Lift of Petersen ----------------
    def voltage_lift_petersen():
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

        # All non-backtracking closed walks of length [5, g-1] -> constraint vectors
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
                        walk_vectors.add(tuple(new_counts))
                    stack.append((next_v, current, new_l, new_counts))

        unique_constraints = set()
        for c in walk_vectors:
            if all(x == 0 for x in c):
                continue
            first_non_zero = -1
            for i in range(15):
                if c[i] != 0:
                    first_non_zero = i
                    break
            if first_non_zero != -1:
                if c[first_non_zero] < 0:
                    c = tuple(-x for x in c)
                unique_constraints.add(c)

        C = np.array(list(unique_constraints), dtype=np.int64)
        if C.size == 0:
            return None
        C = np.mod(C, 10**9)  # keep ints bounded
        n_cons = C.shape[0]

        def build_graph(m, v):
            n = 10 * m
            edges = []
            for u, v_base in pet_edges:
                e_idx = edge_index[(u, v_base)]
                vol = v[e_idx] % m
                for layer in range(m):
                    u_lift = u * m + layer
                    v_lift = v_base * m + (layer + vol) % m
                    if u_lift < v_lift:
                        edges.append((u_lift, v_lift))
                    else:
                        edges.append((v_lift, u_lift))
            return n, edges

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

        def cost(v, m):
            residues = np.mod(C @ v, m)
            return int(np.sum(residues == 0))

        def repair(v, m, time_limit):
            """Surgically fix each violating residue by adjusting one variable.
            Returns (v, cost) after a bounded number of single-variable moves."""
            v = v.copy()
            cur = cost(v, m)
            if cur == 0:
                return v, 0
            it = 0
            while cur > 0 and time.time() < time_limit and it < 4000:
                it += 1
                residues = np.mod(C @ v, m)
                viol = np.where(residues == 0)[0]
                if len(viol) == 0:
                    break
                r = int(viol[random.randrange(len(viol))])
                row = C[r]
                nz = np.nonzero(row % m != 0)[0]
                if len(nz) == 0:
                    break
                j = int(nz[random.randrange(len(nz))])
                coef = int(row[j] % m)
                # need v[j] -> v[j] - coef * inv(coef)  (mod m)
                if coef == 0:
                    continue
                try:
                    inv = pow(coef, m - 2, m)  # m prime
                except Exception:
                    continue
                target = (v[j] - coef * inv) % m
                if target == v[j]:
                    continue
                newv = v.copy()
                newv[j] = target
                nc = cost(newv, m)
                if nc <= cur:
                    v = newv
                    cur = nc
                    if cur == 0:
                        break
            return v, cur

        def solve_voltage(m, max_iters, time_limit):
            v = np.random.randint(1, m, size=15).astype(np.int64)
            cur = cost(v, m)
            best_v = v.copy()
            best_cost = cur
            if best_cost == 0:
                return v, 0

            T = 1.0
            cooling = 0.9998
            min_T = 0.005
            it = 0
            while it < max_iters and time.time() < time_limit:
                it += 1
                j = random.randint(0, 14)
                old_val = v[j]
                v[j] = random.randint(1, m - 1)
                nc = cost(v, m)
                delta = nc - best_cost
                if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                    if nc < best_cost:
                        best_cost = nc
                        best_v = v.copy()
                        if best_cost == 0:
                            return v, 0
                else:
                    v[j] = old_val
                T *= cooling

            # high-precision polish: exact repair from best
            rv, rc = repair(best_v, m, time_limit)
            if rc < best_cost:
                best_v = rv
                best_cost = rc
            return best_v, best_cost

        m = 53
        while m < 2200 and remaining() > 15:
            if is_prime(m):
                # more restarts for small primes (cheaper, more likely to hit 0)
                n_restarts = 30 if m < 120 else (15 if m < 250 else 8)
                t_limit = min(6.0, max(1.5, remaining() / 8.0))
                for restart in range(n_restarts):
                    if remaining() < 12:
                        break
                    v, c = solve_voltage(m, max_iters=6000, time_limit=t_limit)
                    if c == 0:
                        n, edges = build_graph(m, v)
                        gc = check_girth(n, edges, g, deadline - 3)
                        if gc >= g:
                            return (n, edges)
            m += 1
        return None

    # ---------------- Greedy random 3-regular fallback ----------------
    def greedy_growth():
        if k != 3:
            return None
        for n in range(520, 640, 2):
            if remaining() < 12:
                break
            for attempt in range(40):
                if remaining() < 10:
                    break
                half_edges = n * k // 2
                stubs = []
                for i in range(n):
                    stubs.extend([i] * k)
                random.shuffle(stubs)
                edges = set()
                valid = True
                for i in range(0, len(stubs), 2):
                    u, v = stubs[i], stubs[i + 1]
                    if u == v:
                        valid = False
                        break
                    key = (min(u, v), max(u, v))
                    if key in edges:
                        valid = False
                        break
                    edges.add(key)
                if not valid:
                    continue
                elist = list(edges)
                if len(elist) != half_edges:
                    continue
                gc = check_girth(n, elist, g, deadline - 4)
                if gc >= g:
                    return (n, elist)
        return None

    res = voltage_lift_petersen()
    if res is not None:
        return res

    res2 = greedy_growth()
    if res2 is not None:
        return res2

    return (0, [])


if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
