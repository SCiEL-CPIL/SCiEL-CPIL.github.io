"""Path decomposition via greedy cover + max-matching endpoint merges."""
import time
from random import Random
from collections import defaultdict, deque

# ---- instance generator (copied from verifier) ----
def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

# ---- maximum bipartite matching (Hopcroft-Karp, iterative) ----
def _hopcroft_karp(adj, n_left, n_right):
    from math import inf
    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    dist = [0] * n_left

    def bfs():
        dq = deque()
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                dq.append(u)
            else:
                dist[u] = inf
        found = False
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == inf:
                    dist[pu] = dist[u] + 1
                    dq.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = inf
        return False

    matching = 0
    while bfs():
        for u in range(n_left):
            if pair_u[u] == -1 and dfs(u):
                matching += 1
    return pair_u

# ---- one instance ----
def _decompose(edges):
    n = max((max(u, v) for u, v in edges), default=0) + 1
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # 1) greedy path cover (each edge placed once)
    paths = []
    used = set()
    for u, v in edges:
        e = (u, v)
        if e in used:
            continue
        used.add(e)
        # try to extend an existing path at an endpoint
        placed = False
        for i, P in enumerate(paths):
            a, b = P[0], P[-1]
            if (a, v) in adj and (a, v) not in used:
                used.add((a, v)); used.add((v, b))
                paths[i] = [v] + P
                placed = True; break
            if (b, u) in adj and (b, u) not in used:
                used.add((b, u)); used.add((u, a))
                paths[i] = P + [u]
                placed = True; break
        if not placed:
            paths.append([u, v])

    # 2) iterative max-matching merges of fragment endpoints
    for _ in range(200):
        if len(paths) <= 1:
            break
        # vertex -> fragments containing it (a simple path hits a vertex at most once)
        vmap = defaultdict(list)
        for i, P in enumerate(paths):
            for v in P:
                vmap[v].append(i)
        # build candidate joins grouped by vertex
        cand = defaultdict(list)
        for v, lst in vmap.items():
            if len(lst) >= 2:
                cand[v] = lst
        if not cand:
            break
        # auxiliary bipartite graph: left = (frag, endpoint), right = (frag, endpoint)
        node = []            # (frag, endpoint)
        node_index = {}
        adjl = []
        for v, lst in cand.items():
            for i in lst:
                P = paths[i]
                for ep in (0, 1):
                    key = (i, ep)
                    if key not in node_index:
                        node_index[key] = len(node)
                        node.append(key)
                        adjl.append([])
        n_left = len(node)
        n_right = len(node)
        for idx, (i, ep) in enumerate(node):
            P = paths[i]
            other = 1 - ep
            for j in node_index:
                if j[0] == i:
                    continue
                if j[1] == other:
                    continue
                # same vertex is guaranteed by construction only if both in same cand[v]
                # need to check shared vertex; recompute: they are connectable iff share a vertex
                # and are different fragments. We'll test via vertex membership.
                # (cheap: both are in some cand[v]; verify a common vertex exists)
                # Since both were added under possibly different v, verify:
                pass
        # Simpler: build edges by iterating each vertex's fragment list (pairs share that vertex)
        adjl = [[] for _ in range(n_left)]
        for v, lst in cand.items():
            idxs = [node_index[(i, ep)] for i in lst for ep in (0, 1)]
            for a in range(len(idxs)):
                for b in range(len(idxs)):
                    ia = idxs[a]; ib = idxs[b]
                    if ia == ib:
                        continue
                    fa, ea = node[ia]
                    fb, eb = node[ib]
                    if fa == fb:
                        continue
                    if ea == eb:
                        continue
                    adjl[ia].append(ib)
        if not any(adjl):
            break
        pair_u = _hopcroft_karp(adjl, n_left, n_right)
        # collect disjoint matched joins
        joins = []
        used_frag = set()
        for u in range(n_left):
            w = pair_u[u]
            if w == -1:
                continue
            ia, ib = node[u], node[w]
            fa, ea = ia
            fb, eb = ib
            if fa in used_frag or fb in used_frag:
                continue
            if fa == fb:
                continue
            used_frag.add(fa); used_frag.add(fb)
            joins.append((fa, ea, fb, eb))
        if not joins:
            break
        # apply joins
        new_paths = [None] * len(paths)
        done = set()
        for (fa, ea, fb, eb) in joins:
            A = paths[fa]; B = paths[fb]
            if ea == 0:
                A = A[::-1]
            if eb == 0:
                B = B[::-1]
            new_paths[fa] = A + B
            done.add(fa); done.add(fb)
        for i in range(len(paths)):
            if i not in done:
                new_paths[i] = paths[i]
        paths = [p for p in new_paths if p is not None]

    return paths

# ---- validation ----
def _valid(paths, edges):
    eset = set(edges)
    used = set()
    for P in paths:
        if len(P) < 2:
            return False
        if len(set(P)) != len(P):
            return False
        for a, b in zip(P, P[1:]):
            e = tuple(sorted((a, b)))
            if e not in eset or e in used:
                return False
            used.add(e)
    return used == eset

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    for spec in specs:
        edges = _instance(*spec)
        best = None
        # baseline: trivial edge cover (always valid)
        base = [[list(e) for e in edges]]
        # try the algorithm
        try:
            cand = _decompose(edges)
            if _valid(cand, edges):
                best = cand
        except Exception:
            best = None
        if best is None:
            best = base
        results.append(best)
        if time.time() > deadline:
            # fill remaining with trivial covers
            for s2 in specs[len(results):]:
                e2 = _instance(*s2)
                results.append([[list(e) for e in e2]])
            break
    return results
