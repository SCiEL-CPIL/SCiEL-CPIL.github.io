"""Path cover of fixed connected graphs via odd-vertex pairing + edge-disjoint paths."""
import time

def construct():
    t0 = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(n, edges):
        adj = {i: [] for i in range(n)}
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def eulerian_trail(start, adj, n):
        # Hierholzer on multigraph adjacency (list of (neighbor, edge_id))
        # We'll pass a mutable dict of neighbor lists.
        # Return list of vertices in trail order.
        stack = [start]
        trail = []
        while stack:
            v = stack[-1]
            if adj[v]:
                to = adj[v].pop()
                stack.append(to)
            else:
                trail.append(stack.pop())
        trail.reverse()
        return trail

    def min_path_cover(n, edges):
        if not edges:
            return []
        adj = build_adj(n, edges)
        deg = {i: len(adj[i]) for i in range(n)}
        odd = [v for v in range(n) if deg[v] % 2 == 1]

        if len(odd) == 0:
            # Eulerian circuit: pick any vertex with degree>0
            start = next(v for v in range(n) if deg[v] > 0)
            # build mutable adjacency with edge tracking
            m_adj = {i: list(adj[i]) for i in range(n)}
            trail = eulerian_trail(start, m_adj, n)
            return [trail]

        # Pair odd vertices arbitrarily
        pairs = [(odd[i], odd[i + 1]) for i in range(0, len(odd), 2)]

        # Find edge-disjoint paths connecting each pair using max flow.
        # Build a flow network: split each edge into two directed arcs with capacity 1.
        # For each pair (a,b), we need one unit of flow from a to b, all pairs simultaneously.
        # Use a super source S and super sink T: S->a (cap1) for each a in pairs, b->T (cap1) for each b.
        # Actually each pair is a distinct commodity; to make it single-commodity, connect S to each first-vertex and each second-vertex to T, but that allows cross-matching.
        # To force specific pairing, we can do sequential: for each pair, find a path in the residual graph (edge-disjoint by removing used edges).
        # Simpler: sequential shortest path with edge removal (greedy) — may fail but for these small graphs with low odd count, try BFS; if fails, fall back.

        # Greedy: for each pair, BFS for a path using only unused edges; mark used.
        used_edges = set()
        paths = []
        remaining = set(edges)

        # order pairs by something? try as-is; if a pair fails, we can still cover via Eulerian of remainder
        for (a, b) in pairs:
            # BFS from a to b using remaining edges
            from collections import deque
            parent = {a: None}
            q = deque([a])
            found = False
            while q:
                u = q.popleft()
                if u == b:
                    found = True
                    break
                for v in adj[u]:
                    if v in parent:
                        continue
                    e = tuple(sorted((u, v)))
                    if e not in remaining:
                        continue
                    parent[v] = (u, e)
                    q.append(v)
            if not found:
                # cannot find path; break, will handle remainder
                break
            # reconstruct
            path = []
            cur = b
            while cur != a:
                path.append(cur)
                cur, e = parent[cur]
                remaining.discard(e)
            path.append(a)
            path.reverse()
            paths.append(path)

        # Now remaining edges form a graph where all vertices have even degree (since we removed
        # paths whose endpoints were odd vertices, removing a path flips parity of its two endpoints,
        # turning odd->even). So remaining is Eulerian.
        if remaining:
            # find a vertex with remaining degree > 0
            rem_adj = {i: [] for i in range(n)}
            for (u, v) in remaining:
                rem_adj[u].append(v)
                rem_adj[v].append(u)
            start = next(v for v in range(n) if rem_adj[v])
            trail = eulerian_trail(start, rem_adj, n)
            paths.append(trail)

        return paths

    result = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            # fallback: trivial edge-as-path for this and rest
            for s in spec:
                pass
            # just fill with trivial for remaining specs
            break
        edges = instance(*spec)
        n = spec[0]
        cover = min_path_cover(n, edges)
        result.append(cover)

    # If we ran out of time, fill remaining with trivial covers
    while len(result) < 3:
        idx = len(result)
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[idx]
        edges = instance(*spec)
        result.append([list(e) for e in edges])

    return result
