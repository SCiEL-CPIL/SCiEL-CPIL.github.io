import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill for safety

    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(r))

    def seed_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            rr, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][rr]

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

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check(n, adj, g):
        """Certified exact girth check via non-backtracking BFS from every vertex."""
        half_g = g // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def check_girth_circulant(n, S, g):
        """
        Vertex-transitive girth check for circulant graph C_n(S).
        BFS from vertex 0: if any non-tree edge creates a cycle < g, fail.
        Sufficient because vertex-transitivity lets us translate any cycle to contain 0.
        """
        max_depth = (g - 1) // 2
        dist = [-1] * n
        parent = [-1] * n
        dist[0] = 0
        q = deque([0])

        while q:
            u = q.popleft()
            if dist[u] >= max_depth:
                continue
            for v in S:
                w = (u + v) % n
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    parent[w] = u
                    q.append(w)
                else:
                    # Non-tree edge: check cycle length
                    if w == parent[u]:
                        continue
                    cycle_len = dist[u] + dist[w] + 1
                    if cycle_len < g:
                        return False
        return True

    def try_circulant_search(k, g, seed_n, deadline):
        """
        Search for a circulant k-regular graph with girth >= g and n < seed_n.
        For k=5: S = {n/2, a, -a, b, -b} with a, b in [1, n/2-1], a != b.
        Exploits vertex-transitivity for fast single-source girth check.
        """
        if k != 5:
            return None

        # Feasibility: BFS tree from 0 to depth (g-1)//2 must fit in n vertices
        max_depth = (g - 1) // 2
        tree_size = 1 + k * sum((k - 1) ** i for i in range(max_depth))

        start_n = max(moore_bound, tree_size)
        if start_n % 2 != 0:
            start_n += 1
        start_n = max(start_n, 8)

        end_n = min(seed_n, start_n + 800)

        for n in range(start_n, end_n, 2):
            if time.time() > deadline:
                break

            half_n = n // 2
            num_candidates = half_n - 1
            if num_candidates < 2:
                continue

            # For small n, do a full deterministic sweep
            # C(half_n-1, 2) is the total number of (a,b) pairs
            total_pairs = num_candidates * (num_candidates - 1) // 2

            if total_pairs <= 50000:
                # Deterministic full sweep
                for a in range(1, half_n):
                    if time.time() > deadline:
                        break
                    for b in range(1, half_n):
                        if a == b:
                            continue
                        S = (half_n, a, n - a, b, n - b)
                        if len(set(S)) != 5:
                            continue
                        if check_girth_circulant(n, S, g):
                            edges = []
                            for i in range(n):
                                for s in S:
                                    j = (i + s) % n
                                    if i < j:
                                        edges.append((i, j))
                            return n, edges
            else:
                # Random sampling for larger n
                # Scale samples: more for smaller n
                samples = min(2000, max(50, 200000 // n))
                for trial in range(samples):
                    if time.time() > deadline:
                        break
                    a = random.randint(1, half_n - 1)
                    b = random.randint(1, half_n - 1)
                    if a == b:
                        continue
                    S = (half_n, a, n - a, b, n - b)
                    if len(set(S)) != 5:
                        continue
                    if check_girth_circulant(n, S, g):
                        edges = []
                        for i in range(n):
                            for s in S:
                                j = (i + s) % n
                                if i < j:
                                    edges.append((i, j))
                        return n, edges

        return None

    # Build the seed construction first (our guaranteed fallback)
    seed_n, seed_edges = seed_construct(k, g)

    # Try circulant search for a better (smaller n) solution
    result = try_circulant_search(k, g, seed_n, deadline)

    if result is not None:
        n, edges = result
        # Verify with exact girth check before returning
        adj = build_adj(edges, n)
        if exact_girth_check(n, adj, g):
            return n, edges

    # Fallback to seed
    return seed_n, seed_edges
