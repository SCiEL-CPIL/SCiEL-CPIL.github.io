"""Rewritten program: maximum 2-factor extraction (via Hopcroft-Karp matching)
followed by greedy long-cycle packing on the residual, then forest edges."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def canonical(e):
        a, b = e
        return (a, b) if a < b else (b, a)

    def hopcroft_karp(adj, n):
        """Maximum cardinality matching in a bipartite graph.
        adj: dict u -> set(v). Returns matchR: dict v -> u (for matched v)."""
        INF = float('inf')
        matchL = {}
        matchR = {}
        dist = {}

        def bfs():
            q = deque()
            for u in adj:
                if u not in matchL:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = INF
            found = False
            while q:
                u = q.popleft()
                for v in adj[u]:
                    mu = matchR.get(v)
                    if mu is None:
                        found = True
                    elif dist.get(mu, INF) == INF:
                        dist[mu] = dist[u] + 1
                        q.append(mu)
            return found

        def dfs(u):
            for v in adj[u]:
                mu = matchR.get(v)
                if mu is None or (dist.get(mu, INF) == dist[u] + 1 and dfs(mu)):
                    matchL[u] = v
                    matchR[v] = u
                    return True
            dist[u] = INF
            return False

        while bfs():
            for u in list(adj.keys()):
                if u not in matchL:
                    dfs(u)
        return matchR

    def build_2factor(adj):
        """Find a maximum 2-factor (max edge-disjoint cycles) in the graph
        using a maximum matching in the bipartite double cover.
        Returns list of cycles (each a list of distinct vertices, len>=3)."""
        # Build bipartite adjacency: L = {u}, R = {v} with same edges.
        # adj is undirected: adj[u] = set of neighbors.
        # For matching, left node u can match right node v if (u,v) is an edge, u != v.
        bip_adj = {u: set() for u in adj}
        for u in adj:
            for v in adj[u]:
                bip_adj[u].add(v)
        matchR = hopcroft_karp(bip_adj, len(adj))

        # Build 2-factor degree: each matched pair (u,v) contributes edge u-v.
        # In the 2-factor, each vertex has degree 2 (one as L, one as R) if fully matched.
        # Build the 2-factor graph (undirected) from matched edges.
        factor = defaultdict(set)
        for v, u in matchR.items():
            factor[u].add(v)
            factor[v].add(u)

        # Extract cycles from factor (each vertex has degree <= 2).
        cycles = []
        seen_edges = set()
        for u in list(factor.keys()):
            if not factor[u]:
                continue
            # Find a cycle containing u by walking.
            # Since degree <= 2, we can trace.
            # Find a vertex with degree 2 to start a cycle (degree 1 = path end, skip).
            pass
        # Better: trace all cycles.
        visited_v = set()
        for start in list(factor.keys()):
            if start in visited_v:
                continue
            if len(factor[start]) < 2:
                # path end or isolated; not part of a cycle
                continue
            # Trace a cycle starting from start.
            path = [start]
            prev = None
            cur = start
            while True:
                nxts = [x for x in factor[cur] if x != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                if nxt == start:
                    # closed cycle
                    if len(path) >= 3:
                        cycles.append(list(path))
                    break
                if nxt in path:
                    # closed into a cycle (should be start in a 2-factor)
                    idx = path.index(nxt)
                    cyc = path[idx:]
                    if len(cyc) >= 3:
                        cycles.append(list(cyc))
                    break
                path.append(nxt)
                prev, cur = cur, nxt
            # mark vertices in the traced cycle as visited
            for x in path:
                visited_v.add(x)
        return cycles

    def find_long_cycle(adj, remaining, max_starts=12, tries=4):
        """Find a reasonably long simple cycle in the remaining graph."""
        best = None
        if not adj:
            return None
        # Start from high-degree vertices
        verts = sorted(adj.keys(), key=lambda x: (-len(adj[x]), x))
        for start in verts[:max_starts]:
            if len(adj[start]) < 2:
                continue
            for _ in range(tries):
                # randomized DFS to find a cycle through start
                visited = {start}
                path = [start]
                nbr_order = {}
                # Pre-shuffle neighbor lists for this try
                for u in list(adj.keys()):
                    nbrs = list(adj[u])
                    random.shuffle(nbrs)
                    nbr_order[u] = nbrs

                def dfs(u):
                    for v in nbr_order.get(u, []):
                        if v == start and len(path) >= 2:
                            return True
                        if v not in visited:
                            visited.add(v)
                            path.append(v)
                            if dfs(v):
                                return True
                            path.pop()
                            visited.remove(v)
                    return False

                if dfs(start):
                    cyc = list(path)
                    if best is None or len(cyc) > len(best):
                        best = cyc
                    if len(best) >= 25:
                        return best
        return best

    def decompose(edges):
        if not edges:
            return []

        adj = defaultdict(set)
        remaining = set()
        for e in edges:
            c = canonical(e)
            remaining.add(c)
            adj[c[0]].add(c[1])
            adj[c[1]].add(c[0])

        pieces = []

        def remove_edges_of_cycle(cyc):
            for i in range(len(cyc)):
                a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                e = canonical((a, b))
                remaining.discard(e)
                adj[a].discard(b)
                adj[b].discard(a)
                if not adj[a]:
                    adj.pop(a, None)
                if not adj[b]:
                    adj.pop(b, None)

        # Phase 1: extract maximum 2-factor (edge-disjoint cycles).
        if time.time() - t0 < DEADLINE - 30:
            cycles = build_2factor(adj)
            # Validate and remove
            for cyc in cycles:
                if time.time() - t0 > DEADLINE - 10:
                    break
                valid = True
                for i in range(len(cyc)):
                    a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                    if canonical((a, b)) not in remaining:
                        valid = False
                        break
                if valid and len(cyc) >= 3:
                    # ensure all distinct
                    if len(set(cyc)) == len(cyc):
                        pieces.append(list(cyc))
                        remove_edges_of_cycle(cyc)

        # Phase 2: greedy long-cycle packing on residual.
        while remaining:
            if time.time() - t0 > DEADLINE - 8:
                break
            cyc = find_long_cycle(adj, remaining)
            if cyc and len(cyc) >= 3 and len(set(cyc)) == len(cyc):
                valid = True
                for i in range(len(cyc)):
                    a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                    if canonical((a, b)) not in remaining:
                        valid = False
                        break
                if valid:
                    pieces.append(list(cyc))
                    remove_edges_of_cycle(cyc)
                else:
                    break
            else:
                # No more cycles: remaining is a forest.
                for e in list(remaining):
                    pieces.append(list(e))
                    a, b = e
                    adj[a].discard(b)
                    adj[b].discard(a)
                    if not adj[a]:
                        adj.pop(a, None)
                    if not adj[b]:
                        adj.pop(b, None)
                    remaining.discard(e)
                break

        # Phase 3: any leftover edges (shouldn't happen) as single edges.
        for e in list(remaining):
            pieces.append(list(e))
        remaining = set()

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        if time.time() - t0 > DEADLINE - 15:
            results.append([list(e) for e in edges])
        else:
            results.append(decompose(edges))

    return results
# EVOLVE-BLOCK-END
