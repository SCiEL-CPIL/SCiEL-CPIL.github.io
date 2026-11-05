"""Vertex-cover solver: S meets every maximal clique (size>=2) iff V\S is an
independent set, i.e. S is a vertex cover.  We minimize total |S| over three
fixed random graphs using a branch-and-bound minimum vertex cover solver with
repeated randomization + local search, guarded by a wall-clock deadline."""

import random
import time


def _build_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj


def _is_vertex_cover(adj, cover_set):
    for u in range(len(adj)):
        if u in cover_set:
            continue
        # u not in cover -> all its neighbours must be in cover
        for v in adj[u]:
            if v not in cover_set:
                return False
    return True


def _min_vertex_cover(adj, deadline, rng):
    n = len(adj)
    # Fast greedy seed solution (lower bound on cost we can beat)
    best_set = set(_greedy_vc(adj))
    best_cost = len(best_set)

    def _branch(cover, uncovered_edges, cost):
        nonlocal best_set, best_cost
        if time.time() > deadline:
            return
        if cost >= best_cost:
            return
        # pick an uncovered edge
        if not uncovered_edges:
            if cost < best_cost:
                best_cost = cost
                best_set = set(cover)
            return
        # choose edge with max degree sum (heuristic)
        e = max(uncovered_edges, key=lambda uv: len(adj[uv[0]]) + len(adj[uv[1]]))
        u, v = e
        # branch: add u
        new_cover = cover | {u}
        new_uncov = set()
        for a, b in uncovered_edges:
            if a == u or b == u:
                continue
            new_uncov.add((a, b) if a < b else (b, a))
        _branch(new_cover, new_uncov, cost + 1)
        if time.time() > deadline:
            return
        # branch: add v
        new_cover = cover | {v}
        new_uncov = set()
        for a, b in uncovered_edges:
            if a == v or b == v:
                continue
            new_uncov.add((a, b) if a < b else (b, a))
        _branch(new_cover, new_uncov, cost + 1)

    # initial uncovered edges
    init_uncov = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                init_uncov.add((u, v))

    _branch(set(), init_uncov, 0)

    # local search: try removing a vertex from the cover, keep if still a cover
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u in list(best_set):
            cand = best_set - {u}
            if _is_vertex_cover(adj, cand):
                best_set = cand
                best_cost = len(best_set)
                improved = True
    return best_set


def _greedy_vc(adj):
    n = len(adj)
    cover = set()
    remaining = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                remaining.add((u, v))
    while remaining:
        # pick vertex covering the most remaining edges
        cnt = {}
        for a, b in remaining:
            cnt[a] = cnt.get(a, 0) + 1
            cnt[b] = cnt.get(b, 0) + 1
        u = max(cnt, key=lambda x: cnt[x])
        cover.add(u)
        remaining = {(a, b) for (a, b) in remaining if a != u and b != u}
    return cover


def construct():
    deadline = time.time() + 140.0  # safety margin under 150s
    rng = random.Random(12345)
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        adj = _build_graph(n, p, seed)
        # multiple randomized restarts of the B&B to escape bad branch orders
        best = None
        best_cost = None
        while time.time() < deadline:
            r = random.Random(rng.random())
            s = _min_vertex_cover(adj, deadline, r)
            if best_cost is None or len(s) < best_cost:
                best_cost = len(s)
                best = s
            if time.time() > deadline:
                break
        out.append(sorted(best))
    return out
