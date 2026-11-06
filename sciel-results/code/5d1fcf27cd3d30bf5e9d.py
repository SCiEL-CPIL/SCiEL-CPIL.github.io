"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import deque

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def canonical(e):
        a, b = e
        return (a, b) if a < b else (b, a)

    def hopcroft_karp(left, right, adj):
        """Maximum bipartite matching. adj: dict left->list of right. Returns (matchL, matchR)."""
        matchL = {u: -1 for u in left}
        matchR = {v: -1 for v in right}

        def bfs():
            dist = {}
            q = deque()
            for u in left:
                if matchL[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            found = False
            while q:
                u = q.popleft()
                for v in adj.get(u, ()):
                    mu = matchR[v]
                    if mu == -1:
                        found = True
                    elif dist.get(mu, -1) == -1:
                        dist[mu] = dist[u] + 1
                        q.append(mu)
            return dist, found

        def dfs(u):
            for v in adj.get(u, ()):
                mu = matchR[v]
                if mu == -1 or (dist.get(mu, -1) == dist.get(u, -1) + 1 and dfs(mu)):
                    matchL[u] = v
                    matchR[v] = u
                    return True
            dist[u] = -1
            return False

        while True:
            dist, found = bfs()
            if not found:
                break
            for u in left:
                if matchL[u] == -1:
                    dfs(u)
        return matchL, matchR

    def decompose(edges):
        if not edges:
            return []
        remaining = set(canonical(e) for e in edges)
        pieces = []

        def build_adj(rem):
            adj = {}
            for u, v in rem:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            return adj

        def extract_2factor_cycles(rem):
            """Find a large 2-factor (edge-disjoint cycles) via max bipartite matching."""
            adj = build_adj(rem)
            if not adj:
                return []
            vertices = sorted(adj.keys())
            left = vertices
            right = vertices
            # Build bipartite adjacency: u_L -> v_R for edge {u,v}, u != v
            bip = {u: [] for u in left}
            for u in left:
                for v in adj.get(u, ()):
                    if v != u:
                        bip[u].append(v)
            matchL, matchR = hopcroft_karp(left, right, bip)
            # Build directed cycles from matching: u -> matchL[u]
            visited = set()
            cycles = []
            for u in left:
                if u in visited or matchL[u] == -1:
                    continue
                # follow cycle
                cyc = []
                cur = u
                while cur not in visited:
                    visited.add(cur)
                    cyc.append(cur)
                    nxt = matchL[cur]
                    if nxt == -1:
                        break
                    cur = nxt
                if len(cyc) >= 3:
                    cycles.append(cyc)
            return cycles

        def cycle_edges(cyc):
            es = set()
            m = len(cyc)
            for i in range(m):
                es.add(canonical((cyc[i], cyc[(i + 1) % m])))
            return es

        def heuristic_longest_cycle(rem, adj, tries=200):
            """Randomized walk heuristic to find a long simple cycle."""
            if not adj:
                return None
            best = None
            rng = Random(hash(frozenset(rem)) & 0xffffffff)
            for _ in range(tries):
                # pick a random start with degree >= 2
                cand = [u for u in adj if len(adj[u]) >= 2]
                if not cand:
                    return best
                start = rng.choice(cand)
                # random DFS walk with backtracking
                path = [start]
                used = {start}
                node = start
                while True:
                    nbrs = [x for x in adj.get(node, ()) if x not in used]
                    if not nbrs:
                        break
                    # prefer closing to start if possible
                    if start in nbrs and len(path) >= 2:
                        # close
                        if best is None or len(path) > len(best):
                            best = list(path)
                        break
                    # prefer neighbors with higher degree (more options)
                    if len(nbrs) > 1:
                        nbrs.sort(key=lambda x: -len(adj.get(x, ())))
                        # add some randomness
                        if rng.random() < 0.3:
                            node = rng.choice(nbrs)
                        else:
                            node = nbrs[0]
                    else:
                        node = nbrs[0]
                    path.append(node)
                    used.add(node)
                # also try to close
                if start in adj.get(node, ()) and len(path) >= 2:
                    if best is None or len(path) > len(best):
                        best = list(path)
            return best if (best and len(best) >= 3) else None

        def find_cycle_bfs(rem, adj):
            """Find any cycle using BFS on the remaining edges."""
            visited = set()
            for start in sorted(adj.keys()):
                if start in visited:
                    continue
                # BFS
                q = deque([(start, [start])])
                while q:
                    node, path = q.popleft()
                    if node in visited:
                        continue
                    visited.add(node)
                    for nbr in sorted(adj.get(node, ())):
                        if nbr == start and len(path) >= 2:
                            # Found a cycle
                            return path
                        if nbr not in visited:
                            q.append((nbr, path + [nbr]))
            return None

        # Phase 1: 2-factor
        if time.time() - t0 < DEADLINE - 5:
            cyc2 = extract_2factor_cycles(remaining)
            for c in cyc2:
                ces = cycle_edges(c)
                if ces <= remaining:
                    pieces.append(list(c))
                    remaining -= ces

        # Phase 2: heuristic longest-cycle extraction on remainder
        while remaining:
            if time.time() - t0 > DEADLINE - 5:
                break
            adj = build_adj(remaining)
            cyc = heuristic_longest_cycle(remaining, adj, tries=150)
            if cyc and len(cyc) >= 3:
                ces = cycle_edges(cyc)
                if ces <= remaining:
                    pieces.append(list(cyc))
                    remaining -= ces
                    continue
            # Fallback: find any cycle
            cyc = find_cycle_bfs(remaining, adj)
            if cyc and len(cyc) >= 3:
                ces = cycle_edges(cyc)
                if ces <= remaining:
                    pieces.append(list(cyc))
                    remaining -= ces
                    continue
            break

        # Phase 3: local improvement - try to merge adjacent edges into longer cycles
        # Try to find 2-edge paths that can be extended
        improved = True
        while improved and time.time() - t0 < DEADLINE - 3:
            improved = False
            adj = build_adj(remaining)
            if not adj:
                break
            # Find paths of length 2 (u-v-w) where u-w is also an edge
            # This would form a triangle
            for u in sorted(adj.keys()):
                if time.time() - t0 > DEADLINE - 3:
                    break
                for v in sorted(adj.get(u, ())):
                    for w in sorted(adj.get(v, ())):
                        if w != u and w != v and (u, w) in adj.get(v, set()):
                            # Check if edges (u,v), (v,w), (u,w) are all in remaining
                            e1 = canonical((u, v))
                            e2 = canonical((v, w))
                            e3 = canonical((u, w))
                            if e1 in remaining and e2 in remaining and e3 in remaining:
                                # Form a triangle
                                remaining -= {e1, e2, e3}
                                pieces.append([u, v, w])
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            # Try to find 4-cycles
            if not improved and remaining:
                adj = build_adj(remaining)
                for u in sorted(adj.keys()):
                    if time.time() - t0 > DEADLINE - 3:
                        break
                    for v in sorted(adj.get(u, ())):
                        if v == u:
                            continue
                        for w in sorted(adj.get(v, ())):
                            if w == u or w == v:
                                continue
                            for x in sorted(adj.get(w, ())):
                                if x == u or x == v or x == w:
                                    continue
                                e1 = canonical((u, v))
                                e2 = canonical((v, w))
                                e3 = canonical((w, x))
                                e4 = canonical((x, u))
                                if (e1 in remaining and e2 in remaining and
                                    e3 in remaining and e4 in remaining):
                                    remaining -= {e1, e2, e3, e4}
                                    pieces.append([u, v, w, x])
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break

        for e in sorted(remaining):
            pieces.append(list(e))
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() - t0 > DEADLINE - 10:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            edges = instance(*spec)
            results.append(decompose(edges))

    return results
# EVOLVE-BLOCK-END
