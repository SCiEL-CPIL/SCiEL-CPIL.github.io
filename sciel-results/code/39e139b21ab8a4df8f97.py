import time
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140

    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

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

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def check_degree(adj, k):
        return all(len(a) == k for a in adj)

    def girth_of_graph(n, adj, k, limit_check=None):
        """Find the girth (shortest cycle) of a regular graph.
        Returns the girth, or inf if no cycle found within time budget.
        Uses BFS from each vertex tracking (node, prev) to avoid backtracking.
        """
        if not check_degree(adj, k):
            return 0

        best = float('inf')
        max_depth = limit_check if limit_check is not None else g + 2

        for s in range(n):
            if time.time() > deadline:
                break

            dist = {}
            dq = deque()
            for nb in adj[s]:
                state = (nb, s)
                dist[state] = 1
                dq.append(state)

            while dq:
                curr, prev = dq.popleft()
                d = dist[(curr, prev)]

                if d >= max_depth:
                    continue

                if curr == s:
                    cycle_len = d + 1
                    if cycle_len < best:
                        best = cycle_len
                    continue

                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    state = (nb, curr)
                    if state not in dist:
                        dist[state] = d + 1
                        dq.append(state)

        return best

    def validate_graph(n, edges, k, g):
        """Full validation: simple, k-regular, girth >= g."""
        adj = build_adj(n, edges)
        if not check_degree(adj, k):
            return False
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        girth = girth_of_graph(n, adj, k, limit_check=g + 1)
        return girth >= g

    def search_circulant(k, g, n_max, n_min=None):
        if n_min is None:
            n_min = max(2 * (k - 1) ** (g // 2), 244)

        def circulant_girth(n, s, limit):
            steps = [1, -1, s % n, (-s) % n]
            steps = list(set(steps))
            if len(steps) < 4:
                return 0

            for L in range(3, limit):
                def gen(prefix, idx, current_sum):
                    if idx == L:
                        if current_sum % n != 0:
                            return None
                        if prefix[0] == (-prefix[-1]) % n:
                            return None
                        return L
                    for step in steps:
                        if idx > 0 and step == (-prefix[-1]) % n:
                            continue
                        result = gen(prefix + [step], idx + 1, current_sum + step)
                        if result is not None:
                            return result
                    return None

                result = gen([], 0, 0)
                if result is not None:
                    return result

            return float('inf')

        best_n = None
        best_s = None

        for n in range(n_min, n_max + 1, 2):
            if time.time() > deadline:
                break

            s_candidates = set()
            for s in [n // 4, n // 3, (n // 4) + 1, (n // 3) + 1, n // 5, (n // 5) + 1,
                      n // 4 - 1, n // 3 - 1, 7, 11, 13, 17, 19]:
                s = s % n
                if s not in [0, 1, (n - 1) % n, (n - 2) % n, 2]:
                    s_candidates.add(s)
            for _ in range(5):
                s = np.random.randint(3, n - 3)
                s_candidates.add(s % n)

            for s in s_candidates:
                s_mod = s % n
                if s_mod in [0, 1, (n - 1) % n, (n - 2) % n, 2]:
                    continue
                steps_set = {1 % n, (n - 1) % n, s_mod % n, (n - s_mod) % n}
                if len(steps_set) != 4:
                    continue

                girth = circulant_girth(n, s_mod, g + 1)
                if girth >= g:
                    if best_n is None or n < best_n:
                        best_n = n
                        best_s = s_mod
                        if n <= 250:
                            break

            if best_n is not None and best_n <= 250:
                break

        return best_n, best_s

    def make_circulant_edges(n, s):
        edges = []
        for u in range(n):
            for v in [(u + 1) % n, (u - 1) % n, (u + s) % n, (u - s) % n]:
                if u < v:
                    edges.append((u, v))
        return edges

    def count_short_cycles(n, adj, k, max_len):
        """Count all cycles of length < max_len using non-backtracking BFS from each vertex."""
        total = 0
        for s in range(n):
            if time.time() > deadline:
                return total
            # BFS tracking (curr, prev)
            dist = {}
            dq = deque()
            for nb in adj[s]:
                state = (nb, s)
                dist[state] = 1
                dq.append(state)

            while dq:
                curr, prev = dq.popleft()
                d = dist[(curr, prev)]
                if d >= max_len - 1:
                    continue
                if curr == s:
                    total += 1
                    continue
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    state = (nb, curr)
                    if state not in dist:
                        dist[state] = d + 1
                        dq.append(state)
        return total

    def local_search_reduce_cycles(n, adj, k, g, max_iters=5000):
        """Local search: minimize number of short cycles (< g) via edge swaps.
        Uses a smooth surrogate: total count of cycles of length 3..g-1."""
        if time.time() > deadline - 15:
            return None

        edge_set = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_set.add((u, v))

        # Initial cost: count of short cycles
        max_len = g
        current_cost = count_short_cycles(n, adj, k, max_len)
        best_cost = current_cost
        best_edge_set = set(edge_set)

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            edges_list = list(edge_set)
            if len(edges_list) < 2:
                break
            i1 = np.random.randint(len(edges_list))
            i2 = np.random.randint(len(edges_list))
            if i1 == i2:
                continue

            u1, v1 = edges_list[i1]
            u2, v2 = edges_list[i2]

            for (a, b, c, d) in [(u1, u2, v1, v2), (u1, v2, v1, u2)]:
                if a == b or c == d:
                    continue
                if (min(a, b), max(a, b)) in edge_set or (min(c, d), max(c, d)) in edge_set:
                    continue

                # Remove old edges
                adj[u1].discard(v1)
                adj[v1].discard(u1)
                adj[u2].discard(v2)
                adj[v2].discard(u2)
                edge_set.discard((min(u1, v1), max(u1, v1)))
                edge_set.discard((min(u2, v2), max(u2, v2)))

                # Add new edges
                adj[a].add(b)
                adj[b].add(a)
                adj[c].add(d)
                adj[d].add(c)
                edge_set.add((min(a, b), max(a, b)))
                edge_set.add((min(c, d), max(c, d)))

                if not check_degree(adj, k):
                    # Revert
                    adj[a].discard(b)
                    adj[b].discard(a)
                    adj[c].discard(d)
                    adj[d].discard(c)
                    edge_set.discard((min(a, b), max(a, b)))
                    edge_set.discard((min(c, d), max(c, d)))
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    edge_set.add((min(u1, v1), max(u1, v1)))
                    edge_set.add((min(u2, v2), max(u2, v2)))
                    continue

                # Evaluate new cost
                new_cost = count_short_cycles(n, adj, k, max_len)
                if new_cost < current_cost:
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_edge_set = set(edge_set)
                        if new_cost == 0:
                            # Girth >= g achieved
                            return list(best_edge_set)
                else:
                    # Revert
                    adj[a].discard(b)
                    adj[b].discard(a)
                    adj[c].discard(d)
                    adj[d].discard(c)
                    edge_set.discard((min(a, b), max(a, b)))
                    edge_set.discard((min(c, d), max(c, d)))
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    edge_set.add((min(u1, v1), max(u1, v1)))
                    edge_set.add((min(u2, v2), max(u2, v2)))

        # If we found a better edge set with cost 0, return it
        if best_cost == 0:
            return list(best_edge_set)
        return None

    # Build LUW graph
    n_luw, edges_luw = luw_construct(k, g)

    # Try circulant search if we have time and LUW is large enough
    if time.time() < deadline - 30 and n_luw > 244:
        n_circ, s_circ = search_circulant(k, g, n_luw - 1, 244)
        if n_circ is not None:
            edges_circ = make_circulant_edges(n_circ, s_circ)
            if validate_graph(n_circ, edges_circ, k, g):
                return n_circ, edges_circ

    # Validate LUW graph
    adj_luw = build_adj(n_luw, edges_luw)
    if check_degree(adj_luw, k):
        girth_luw = girth_of_graph(n_luw, adj_luw, k, limit_check=g + 1)
        if girth_luw >= g:
            return n_luw, edges_luw

    # Try local search to reduce short cycles (smooth surrogate objective)
    improved_edges = local_search_reduce_cycles(n_luw, adj_luw, k, g)
    if improved_edges is not None:
        # Validate the improved graph
        if validate_graph(n_luw, improved_edges, k, g):
            return n_luw, improved_edges

    # Fallback: return LUW graph as-is
    return n_luw, edges_luw
