"""Rewritten: heuristic 2-opt edge-swap girth maximization on a 3-regular
graph, with an explicit deadline guard. Different family from the seed's
explicit algebraic (LUW) construction: we start from a small random 3-regular
graph and use local edge swaps to eliminate short cycles, driving the girth
up toward 16. Consistent representation: a set of canonical edges."""
import time
import random
from collections import deque

_DEADLINE = 150.0   # evaluator kills at 150s
_MARGIN = 12.0      # stop this much before the kill


def _now():
    return time.time()


def _build_random_3regular(n, rng):
    """Random 3-regular simple graph on n vertices (n even, n>=4)."""
    stubs = []
    for v in range(n):
        stubs += [v] * 3
    for _ in range(2000):
        rng.shuffle(stubs)
        edges = set()
        ok = True
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b:
                ok = False
                break
            key = (min(a, b), max(a, b))
            if key in edges:
                ok = False
                break
            edges.add(key)
        if ok:
            return edges
    return None


def _neighbours(edges, n):
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _girth_nonbacktracking(adj):
    """Exact girth via BFS of non-backtracking walks from each vertex."""
    n = len(adj)
    best = 10 ** 9
    for start in range(n):
        dist = {}
        dq = deque()
        for nb in adj[start]:
            dist[(nb, start)] = 1
            dq.append((nb, start))
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d + 1 >= best:
                continue
            for nb in adj[v]:
                if nb == prev:
                    continue
                if nb == start:
                    if d + 1 < best:
                        best = d + 1
                    continue
                key = (nb, v)
                if key in dist:
                    if d + 1 < best:
                        best = d + 1
                    continue
                dist[key] = d + 1
                dq.append((nb, v))
    return best


def _find_short_cycle(adj, girth):
    """Return list of vertices forming a cycle of length == girth, or None."""
    n = len(adj)
    for start in range(n):
        dist = {}
        parent = {}
        dq = deque()
        for nb in adj[start]:
            dist[(nb, start)] = 1
            parent[(nb, start)] = (start, None)
            dq.append((nb, start))
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d + 1 > girth:
                continue
            for nb in adj[v]:
                if nb == prev:
                    continue
                if nb == start:
                    if d + 1 == girth:
                        path = [(v, prev)]
                        cur = (v, prev)
                        while parent[cur][1] is not None:
                            cur = parent[cur]
                            path.append(cur)
                        verts = [start]
                        for (vv, pp) in reversed(path):
                            verts.append(vv)
                        return verts
                    continue
                key = (nb, v)
                if key in dist:
                    if d + 1 == girth:
                        p1 = [(v, prev)]
                        cur = (v, prev)
                        while parent[cur][1] is not None:
                            cur = parent[cur]
                            p1.append(cur)
                        p2 = [(nb, v)]
                        cur2 = (nb, v)
                        while parent[cur2][1] is not None:
                            cur2 = parent[cur2]
                            p2.append(cur2)
                        verts1 = [start]
                        for (vv, pp) in reversed(p1):
                            verts1.append(vv)
                        verts2 = []
                        for (vv, pp) in reversed(p2):
                            verts2.append(vv)
                        verts2.append(start)
                        full = verts1 + verts2[:-1]
                        return full
                    continue
                dist[key] = d + 1
                parent[key] = (v, prev)
                dq.append((nb, v))
    return None


def _swap_candidates(e1, e2):
    a, b = e1
    c, d = e2
    for n1, n2 in [((a, c), (b, d)), ((a, d), (b, c))]:
        k1 = (min(n1[0], n1[1]), max(n1[0], n1[1]))
        k2 = (min(n2[0], n2[1]), max(n2[0], n2[1]))
        if k1[0] == k1[1] or k2[0] == k2[1] or k1 == k2:
            continue
        yield k1, k2


def construct(k, g):
    t0 = _now()
    deadline = t0 + _DEADLINE - _MARGIN
    rng = random.Random(12345)

    if k != 3:
        # Best-effort generic random k-regular (only robust for small k).
        n = 2 * k
        edges = _build_random_3regular(n, rng) if k == 3 else None
        return n, (list(edges) if edges else [])

    n = 96
    edges = _build_random_3regular(n, rng)
    if edges is None:
        return n, []

    best_edges = set(edges)
    best_girth = _girth_nonbacktracking(_neighbours(edges, n))

    def apply_swap(edges, e1, e2, res):
        edges = set(edges)
        edges.discard(e1)
        edges.discard(e2)
        edges.add(res[0])
        edges.add(res[1])
        return edges

    def try_random_swap(edges, n):
        elist = list(edges)
        if len(elist) < 2:
            return edges
        for _ in range(60):
            if _now() > deadline:
                return edges
            e1, e2 = rng.sample(elist, 2)
            for res in _swap_candidates(e1, e2):
                k1, k2 = res
                if k1 in edges or k2 in edges:
                    continue
                return apply_swap(edges, e1, e2, res)
        return edges

    while _now() < deadline:
        adj = _neighbours(edges, n)
        girth = _girth_nonbacktracking(adj)
        if girth >= g:
            return n, list(edges)
        if girth > best_girth:
            best_girth = girth
            best_edges = set(edges)

        cyc = _find_short_cycle(adj, girth)
        if cyc is None or len(cyc) < 3:
            edges = try_random_swap(edges, n)
            continue

        m = len(cyc)
        cycle_edges = []
        for i in range(m):
            u = cyc[i]
            v = cyc[(i + 1) % m]
            cycle_edges.append((min(u, v), max(u, v)))
        i1 = rng.randrange(m)
        e1 = cycle_edges[i1]
        elist = list(edges)
        done = False
        for _ in range(40):
            if _now() > deadline:
                break
            e2 = rng.choice([e for e in elist if e != e1])
            for res in _swap_candidates(e1, e2):
                k1, k2 = res
                if k1 in edges or k2 in edges:
                    continue
                new_edges = apply_swap(edges, e1, e2, res)
                # accept if it doesn't create a shorter girth
                new_girth = _girth_nonbacktracking(_neighbours(new_edges, n))
                if new_girth >= girth:
                    edges = new_edges
                    done = True
                    break
            if done:
                break
        if not done:
            edges = try_random_swap(edges, n)

    return n, list(best_edges)
