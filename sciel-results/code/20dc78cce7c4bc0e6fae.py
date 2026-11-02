import time
import random
from collections import deque
from itertools import combinations

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    def luw_construction():
        """Construct the Lazebnik-Ustimenko-Woldar graph CD(d, q)."""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    def nonbacktracking_girth(adj, g):
        """Return True if the graph has a cycle shorter than g."""
        n = len(adj)
        for s in range(n):
            for a in adj[s]:
                dist = {}
                dist[(a, s)] = 1
                q = deque([(a, s)])
                while q:
                    v, p = q.popleft()
                    d = dist[(v, p)]
                    if d >= g - 1:
                        continue
                    for w in adj[v]:
                        if w == p:
                            continue
                        new_state = (w, v)
                        if new_state not in dist:
                            dist[new_state] = d + 1
                            if w == s and v != a:
                                if d + 1 < g:
                                    return True
                            q.append(new_state)
        return False

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    def random_k_regular(n, k, seed=None):
        """Generate a random k-regular graph on n vertices using configuration model."""
        if n * k % 2 != 0:
            return None
        rng = random.Random(seed)
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        rng.shuffle(half_edges)
        edges = set()
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:
                return None
            key = (min(u, v), max(u, v))
            if key in edges:
                return None
            edges.add(key)
            adj[u].append(v)
            adj[v].append(u)
        return sorted([list(e) for e in edges])

    def try_local_search(n_target, k, g, deadline):
        """Try to find a k-regular graph on n_target vertices with girth >= g."""
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            edges = random_k_regular(n_target, k, seed=attempt)
            if edges is None:
                continue
            adj = build_adj(edges, n_target)
            if not nonbacktracking_girth(adj, g):
                return n_target, edges
        return None

    # First, get the LUW construction as baseline
    n_luw, edges_luw = luw_construction()
    best_n, best_edges = n_luw, edges_luw

    # Moore bound for (4,10): 2 * sum_{i=0}^{4} 3^i = 2*(1+3+9+27+81) = 2*121 = 242
    moore_bound = 2 * sum(3**i for i in range(g // 2))
    search_n = moore_bound
    if search_n % 2 != 0:
        search_n += 1

    # Try a few candidate sizes
    candidates = []
    for n_try in range(search_n, best_n, 2):
        candidates.append(n_try)
    for n_try in [250, 260, 280, 300, 310, 318]:
        if search_n <= n_try < best_n:
            candidates.append(n_try)
    candidates = sorted(set(candidates))

    max_candidates = 20
    if len(candidates) > max_candidates:
        candidates = candidates[::max(1, len(candidates) // max_candidates)][:max_candidates]

    for n_try in candidates:
        if time.time() > deadline - 10:
            break
        result = try_local_search(n_try, k, g, deadline)
        if result is not None:
            n_found, edges_found = result
            if n_found < best_n:
                best_n, best_edges = n_found, edges_found
                break

    return best_n, best_edges
