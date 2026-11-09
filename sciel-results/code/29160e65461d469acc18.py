import time
import random
import math

def construct():
    n = 15
    deadline = time.time() + 58.0

    def is_valid_fast(adj, deg):
        """Check: for every edge (u,v), deg-sets of N(u)\\{v} and N(v)\\{u} are disjoint."""
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    continue
                degs_u = set()
                m = adj[u] & ~(1 << v)
                while m:
                    lsb = m & -m
                    degs_u.add(deg[lsb.bit_length() - 1])
                    m ^= lsb
                if not degs_u:
                    continue
                m = adj[v] & ~(1 << u)
                while m:
                    lsb = m & -m
                    if deg[lsb.bit_length() - 1] in degs_u:
                        return False
                    m ^= lsb
        return True

    def greedy_add(adj, deg, edges, end_time):
        """Greedily add missing edges that maintain validity."""
        improved = True
        while improved and time.time() < end_time:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        continue
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    deg[u] += 1
                    deg[v] += 1
                    if is_valid_fast(adj, deg):
                        edges.append([u, v])
                        improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        deg[u] -= 1
                        deg[v] -= 1
        return edges

    def make_complete_bipartite(a_size):
        """Create K_{a_size, n-a_size}."""
        adj = [0] * n
        deg = [0] * n
        edges = []
        for u in range(a_size):
            for v in range(a_size, n):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])
        return adj, deg, edges

    best_edges = []
    best_count = 0

    # Strategy 1: K_{7,8} + greedy add
    adj, deg, edges = make_complete_bipartite(7)
    edges = greedy_add(adj, deg, edges, deadline - 35)
    if len(edges) > best_count:
        best_count = len(edges)
        best_edges = edges[:]

    # Strategy 2: Local search (remove-then-add)
    search_end = deadline - 15
    while time.time() < search_end:
        a, d, e = adj[:], deg[:], edges[:]
        if not e:
            break
        idx = random.randint(0, len(e) - 1)
        u, v = e[idx]
        a[u] &= ~(1 << v)
        a[v] &= ~(1 << u)
        d[u] -= 1
        d[v] -= 1
        e.pop(idx)
        e = greedy_add(a, d, e, min(time.time() + 0.4, search_end))
        if len(e) > best_count:
            best_count = len(e)
            best_edges = e[:]
            adj, deg, edges = a, d, e

    # Strategy 3: Try other bipartition sizes
    for sz in [5, 6, 8, 9, 10]:
        if time.time() > deadline - 8:
            break
        a, d, e = make_complete_bipartite(sz)
        e = greedy_add(a, d, e, deadline - 8)
        if len(e) > best_count:
            best_count = len(e)
            best_edges = e[:]

    # Strategy 4: Simulated annealing
    if time.time() < deadline - 5:
        a, d, e = adj[:], deg[:], edges[:]
        T = 1.0
        T_min = 0.05
        cooling = 0.999
        while time.time() < deadline - 2 and T > T_min:
            if random.random() < 0.65:
                missing = [(u, v) for u in range(n) for v in range(u + 1, n)
                          if not (a[u] >> v) & 1]
                if not missing:
                    break
                u, v = random.choice(missing)
                a[u] |= (1 << v)
                a[v] |= (1 << u)
                d[u] += 1
                d[v] += 1
                e.append([u, v])
                if is_valid_fast(a, d):
                    if len(e) > best_count:
                        best_count = len(e)
                        best_edges = e[:]
                else:
                    if random.random() >= math.exp(-2.0 / T):
                        a[u] &= ~(1 << v)
                        a[v] &= ~(1 << u)
                        d[u] -= 1
                        d[v] -= 1
                        e.pop()
            else:
                if not e:
                    break
                idx = random.randint(0, len(e) - 1)
                u, v = e[idx]
                a[u] &= ~(1 << v)
                a[v] &= ~(1 << u)
                d[u] -= 1
                d[v] -= 1
                e.pop(idx)
            T *= cooling

    # Final greedy polish
    if time.time() < deadline - 1:
        a = [0] * n
        d = [0] * n
        for u, v in best_edges:
            a[u] |= (1 << v)
            a[v] |= (1 << u)
            d[u] += 1
            d[v] += 1
        e = best_edges[:]
        e = greedy_add(a, d, e, deadline - 0.5)
        if len(e) > best_count:
            best_count = len(e)
            best_edges = e[:]

    out = [[int(u), int(v)] for u, v in best_edges]
    return out
