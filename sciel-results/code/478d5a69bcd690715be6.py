import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

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
        # Tree size = 1 + k*sum_{i=0}^{(g-1)//2 - 1} (k-1)^i
        # For k=5, g=9: 1 + 5*(1+4+16) = 1+5*21 = 106 at depth 3
        # At depth 4: 1+5+20+80+320 = 426
        # If n < tree_size_at_max_depth, collision is forced → infeasible
        max_depth = (g - 1) // 2
        tree_size = 1
        for i in range(max_depth):
            tree_size += k * (k - 1) ** i
        # Actually the tree has 1 + k + k(k-1) + ... up to depth max_depth
        # = 1 + k * sum_{i=0}^{max_depth-1} (k-1)^i
        tree_size = 1 + k * sum((k - 1) ** i for i in range(max_depth))

        start_n = max(moore_bound, tree_size)
        if start_n % 2 != 0:
            start_n += 1
        # Need at least 2 distinct values in [1, n/2-1], so n/2-1 >= 2 => n >= 6
        start_n = max(start_n, 8)

        end_n = min(seed_n, start_n + 800)  # search up to 800 values above Moore bound

        # Precompute candidate pool for speed
        for n in range(start_n, end_n, 2):  # n must be even for k=5
            if time.time() > deadline:
                break

            half_n = n // 2
            # a, b in [1, half_n - 1], a != b
            # Number of valid pairs: C(half_n - 1, 2)
            num_candidates = half_n - 1
            if num_candidates < 2:
                continue

            # Try random samples; scale with available time
            # Budget per n: ~0.05s for the girth checks
            # Each check is O(n * k) ≈ O(n*5) ops
            # For n ~ 500, that's ~2500 ops, very fast
            # So we can try many samples per n

            # Adaptive: more samples for smaller n (cheaper checks)
            samples = min(200, max(20, 50000 // n))

            # Use a fixed-seed approach for reproducibility within a run,
            # but random is fine for search
            for trial in range(samples):
                if time.time() > deadline:
                    break

                # Pick a, b from [1, half_n - 1]
                # Ensure a != b and a + b != n (automatic since a,b < n/2)
                a = random.randint(1, half_n - 1)
                b = random.randint(1, half_n - 1)
                if a == b:
                    continue
                # Also ensure {a, -a, b, -b, n/2} are all distinct mod n
                # -a mod n = n - a, -b mod n = n - b
                # Need: a != b, a != n-b (i.e., a+b != n, true since a,b < n/2)
                #       n-a != n-b (i.e., a != b, already checked)
                #       a != n/2, b != n/2 (true since a,b < n/2)
                #       n-a != n/2 (i.e., a != n/2, true)
                # So just a != b suffices.

                S = (half_n, a, n - a, b, n - b)
                if len(set(S)) != 5:
                    continue

                if check_girth_circulant(n, S, g):
                    # Found valid graph! Build edge list.
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
