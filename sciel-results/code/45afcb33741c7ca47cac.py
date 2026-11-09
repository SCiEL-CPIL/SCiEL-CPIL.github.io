import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.monotonic()
    deadline = start_time + 115.0  # safety margin for 120s limit

    def time_left(margin=5.0):
        return deadline - time.monotonic() - margin

    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if d + 5 < g:
            d += 1
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def girth_ok(adj, n, g):
        """Fast non-backtracking girth check from every vertex."""
        for s in range(n):
            seen = set()
            dq = deque()
            for w in adj[s]:
                st = (w, s)
                seen.add(st)
                dq.append((st, 1))
            while dq:
                (v, p), d = dq.popleft()
                if d >= g:
                    continue
                for w in adj[v]:
                    if w == p:
                        continue
                    if w == s:
                        return False  # found cycle of length d+1 < g
                    st = (w, v)
                    if st not in seen:
                        seen.add(st)
                        dq.append((st, d + 1))
        return True

    def count_short_cycles(adj, n, g):
        """Count cycles of length < g using BFS from each vertex."""
        half = (g - 1) // 2
        count = 0
        for start in range(n):
            parent = [-1] * n
            depth = [-1] * n
            parent[start] = -2
            depth[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if depth[node] >= half:
                    continue
                for nb in adj[node]:
                    if depth[nb] == -1:
                        depth[nb] = depth[node] + 1
                        parent[nb] = node
                        queue.append(nb)
                    elif parent[node] != nb and parent[nb] != node:
                        d1, d2 = depth[node], depth[nb]
                        u, v = node, nb
                        while u != v:
                            if d1 > d2:
                                u = parent[u]
                                d1 -= 1
                            elif d2 > d1:
                                v = parent[v]
                                d2 -= 1
                            else:
                                u = parent[u]
                                v = parent[v]
                                d1 -= 1
                                d2 -= 1
                        cycle_len = depth[node] + depth[nb] - 2 * d1 + 1
                        if cycle_len < g:
                            count += 1
        return count

    def random_regular_graph(n, k, rng):
        if n * k % 2 != 0:
            return None
        for _ in range(100):
            stubs = [i for i in range(n) for _ in range(k)]
            rng.shuffle(stubs)
            edges = []
            seen = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in seen:
                    valid = False
                    break
                seen.add(key)
                edges.append((u, v))
            if valid:
                return edges
        return None

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def cayley_zq_z2(q, a, b):
        """Cayley graph on Z_q x Z_2 with generators +/-a, +/-b, (0,1)."""
        n = 2 * q
        adj = [[] for _ in range(n)]
        edges = []
        for i in range(q):
            for t in (0, 1):
                u = 2 * i + t
                for (di, dt) in ((a, 0), (-a, 0), (b, 0), (-b, 0), (0, 1)):
                    v = 2 * ((i + di) % q) + ((t + dt) & 1)
                    if u < v:
                        adj[u].append(v)
                        adj[v].append(u)
                        edges.append((u, v))
        return n, adj, edges

    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        d = 5
        while d * d <= x:
            if x % d == 0 or x % (d + 2) == 0:
                return False
            d += 6
        return True

    def next_prime(x):
        x = max(2, x)
        while not is_prime(x):
            x += 1
        return x

    # Build LUW baseline
    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # Moore bound
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(r))

    rng = random.Random(42)

    # Phase 1: Try Cayley graphs on Z_q x Z_2 for small n
    lo_q = max(2, (moore_bound + 1) // 2)
    hi_q = best_n // 2

    pairs = []
    for a in range(1, 12):
        for b in range(1, 12):
            if b != a and b != q - a and a != 0 and b != 0:
                pairs.append((a, b))
    # Deduplicate
    seen_pairs = set()
    unique_pairs = []
    for p in pairs:
        if p not in seen_pairs:
            seen_pairs.add(p)
            unique_pairs.append(p)
    pairs = unique_pairs

    q = lo_q
    while q <= hi_q and time_left(3.0) > 0:
        q = next_prime(q)
        for (a, b) in pairs:
            if q <= max(a, b):
                continue
            if time_left(2.0) <= 0:
                break
            n, adj, edges = cayley_zq_z2(q, a % q, b % q)
            if n < best_n and girth_ok(adj, n, g):
                best_n = n
                best_edges = edges
        q += 1

    # Phase 2: Random regular graphs with simulated annealing
    for n in range(max(moore_bound, k + 1), best_n):
        if time_left(10.0) < 0:
            break
        if n * k % 2 != 0:
            continue

        found = False
        for attempt in range(50):
            if time_left(8.0) < 0:
                break

            edges = random_regular_graph(n, k, rng)
            if edges is None:
                continue
            adj = build_adj(n, edges)

            if girth_ok(adj, n, g):
                best_n = n
                best_edges = edges
                found = True
                break

            # Simulated annealing to minimize short cycles
            current_cycles = count_short_cycles(adj, n, g)
            best_cycles = current_cycles
            best_adj = [lst[:] for lst in adj]
            temperature = 10.0
            it = 0
            max_it = 3000

            while it < max_it and time_left(3.0) > 0:
                it += 1
                # Try edge swap
                a_idx = rng.randint(0, n - 1)
                if len(adj[a_idx]) < 2:
                    continue
                b_idx = rng.choice(adj[a_idx])
                c_idx = rng.randint(0, n - 1)
                if c_idx == a_idx or c_idx == b_idx:
                    continue
                if len(adj[c_idx]) < 2:
                    continue
                d_idx = rng.choice(adj[c_idx])
                if d_idx == a_idx or d_idx == b_idx:
                    continue
                if c_idx in adj[a_idx] or d_idx in adj[b_idx]:
                    continue

                # Remove old edges
                adj[a_idx].remove(b_idx)
                adj[b_idx].remove(a_idx)
                adj[c_idx].remove(d_idx)
                adj[d_idx].remove(c_idx)
                # Add new edges
                adj[a_idx].append(c_idx)
                adj[c_idx].append(a_idx)
                adj[b_idx].append(d_idx)
                adj[d_idx].append(b_idx)

                new_cycles = count_short_cycles(adj, n, g)
                delta = new_cycles - current_cycles
                if delta <= 0 or rng.random() < pow(2.718281828, -delta / temperature):
                    current_cycles = new_cycles
                    if new_cycles < best_cycles:
                        best_cycles = new_cycles
                        best_adj = [lst[:] for lst in adj]
                else:
                    # Revert
                    adj[a_idx].remove(c_idx)
                    adj[c_idx].remove(a_idx)
                    adj[b_idx].remove(d_idx)
                    adj[d_idx].remove(b_idx)
                    adj[a_idx].append(b_idx)
                    adj[b_idx].append(a_idx)
                    adj[c_idx].append(d_idx)
                    adj[d_idx].append(c_idx)

                temperature *= 0.995

            if best_cycles == 0:
                best_n = n
                best_edges = [(min(u, v), max(u, v)) for u in range(n) for v in best_adj[u] if u < v]
                found = True
                break

        if found:
            break

    return best_n, best_edges
