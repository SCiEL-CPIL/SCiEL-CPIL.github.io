"""Path decomposition of three fixed connected graphs (Gallai's conjecture)."""
import time
import heapq
from random import Random
from collections import deque

_START = time.time()
_DEADLINE = 120.0
_MARGIN = 8.0


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _START)


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _bfs_path(adj, start, target, used):
    if start == target:
        return [start]
    prev = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if (u, v) in used or (v, u) in used:
                continue
            if v in prev:
                continue
            prev[v] = u
            if v == target:
                path = [v]
                while path[-1] != start:
                    path.append(prev[path[-1]])
                return path[::-1]
            q.append(v)
    return None


def _min_path(adj, start, target, used):
    best = None
    best_len = None
    prev = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if (u, v) in used or (v, u) in used:
                continue
            if v in prev:
                continue
            prev[v] = u
            if v == target:
                path = [v]
                while path[-1] != start:
                    path.append(prev[path[-1]])
                path = path[::-1]
                if best_len is None or len(path) < best_len:
                    best = path
                    best_len = len(path)
            q.append(v)
    return best


def _dijkstra(adj, start, target, used):
    dist = {start: 0}
    prev = {}
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')):
            continue
        if u == target:
            break
        for v in adj[u]:
            if (u, v) in used or (v, u) in used:
                continue
            nd = d + 1
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if target not in dist:
        return None
    path = [target]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1]


def _build_paths(adj, edges, odd, even, n):
    odd = sorted(odd)
    even = sorted(even)
    m = len(odd)
    if m % 2 == 1:
        if even:
            pairs = list(zip(odd[:-1], odd[1::2]))
            pairs.append((odd[-1], even[0]))
        else:
            pairs = list(zip(odd[:-1], odd[1::2]))
            pairs.append((odd[-1], odd[-1]))
    else:
        pairs = list(zip(odd[:-1], odd[1::2]))
    if not pairs:
        return []

    # Try a few pairings (identity + a few random shuffles), keep the best.
    rng = Random(12345)
    candidate_orders = [list(range(m))]
    for _ in range(20):
        c = list(range(m))
        rng.shuffle(c)
        candidate_orders.append(c)

    def evaluate(order):
        if len(order) % 2 == 1:
            # order has odd vertices only (m odd). even vertex is appended.
            perm = list(order)
            perm.append(-1)  # placeholder for even vertex
            pairs = []
            for i in range(0, len(perm), 2):
                a = odd[perm[i]]
                b = even[0] if perm[i + 1] == -1 else odd[perm[i + 1]]
                pairs.append((a, b))
        else:
            perm = list(order)
            pairs = []
            for i in range(0, len(perm), 2):
                pairs.append((odd[perm[i]], odd[perm[i + 1]]))
        return pairs

    best_pairs = None
    best_cost = None
    for order in candidate_orders:
        pairs = evaluate(order)
        used = set()
        total = 0
        ok = True
        for i, (a, b) in enumerate(pairs):
            if a == b:
                # start and end same: need a cycle through a; build a path a..a via two edges
                # Simpler: treat as path from a to a = [a] only if no edges remain for it.
                # We'll handle by building a->a as a closed walk; but path must be simple.
                # Fallback: skip (shouldn't happen if we picked distinct).
                # Build a self-loop-free path: find any edge from a, go out and back not allowed (simple).
                # Instead, just create [a] and rely on edges being covered elsewhere? No.
                # Handle: find neighbor, build path a - x - ... - a impossible simple.
                # So this pairing is invalid; mark high cost.
                ok = False
                break
            nxt = pairs[(i + 1) % len(pairs)][0]
            # path a -> b, threading through nxt
            p1 = _min_path(adj, a, nxt, used)
            if p1 is None:
                ok = False
                break
            for u, v in zip(p1, p1[1:]):
                used.add((min(u, v), max(u, v)))
            p2 = _min_path(adj, nxt, b, used)
            if p2 is None:
                ok = False
                break
            for u, v in zip(p2, p2[1:]):
                used.add((min(u, v), max(u, v)))
            path = p1 + p2[1:]
            total += len(path)
        if not ok:
            continue
        if best_cost is None or total < best_cost:
            best_cost = total
            best_pairs = pairs
    if best_pairs is None:
        best_pairs = evaluate(candidate_orders[0])

    used = set()
    paths = []
    for i, (a, b) in enumerate(best_pairs):
        if a == b:
            # Build a simple cycle through a using two disjoint a-neighbor paths meeting at a.
            # Find two neighbors and a path between them not through a.
            neigh = [v for v in adj[a] if (a, v) not in used and (v, a) not in used]
            found = None
            for x in neigh:
                for y in neigh:
                    if x == y:
                        continue
                    # path x -> y avoiding a
                    adj2 = {u: [v for v in adj[u] if v != a] for u in adj}
                    p = _min_path(adj2, x, y, used)
                    if p is not None:
                        found = (x, y, p)
                        break
                if found:
                    break
            if found:
                x, y, p = found
                path = [a] + p + [a]
                for u, v in zip(path, path[1:]):
                    used.add((min(u, v), max(u, v)))
                paths.append(path)
            else:
                # last resort: single vertex (should not occur)
                paths.append([a])
            continue
        nxt = best_pairs[(i + 1) % len(best_pairs)][0]
        p1 = _min_path(adj, a, nxt, used)
        for u, v in zip(p1, p1[1:]):
            used.add((min(u, v), max(u, v)))
        p2 = _min_path(adj, nxt, b, used)
        for u, v in zip(p2, p2[1:]):
            used.add((min(u, v), max(u, v)))
        paths.append(p1 + p2[1:])
    return paths


def _solve_instance(n, p, seed):
    edges = _instance(n, p, seed)
    adj = {i: [] for i in range(n)}
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    deg = {i: len(adj[i]) for i in range(n)}
    odd = [i for i in range(n) if deg[i] % 2 == 1]
    even = [i for i in range(n) if deg[i] % 2 == 0]
    paths = _build_paths(adj, edges, odd, even, n)
    return paths


def _is_valid(paths, edges, n):
    seen = set()
    for path in paths:
        if len(path) < 1:
            return False
        for i in range(len(path)):
            if not (0 <= path[i] < n):
                return False
        for u, v in zip(path, path[1:]):
            e = (min(u, v), max(u, v))
            if e in seen:
                return False
            if e not in edges:
                return False
            seen.add(e)
        if len(set(path)) != len(path):
            return False
    return seen == edges


def construct():
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    result = []
    for (n, p, seed) in specs:
        if _time_left() <= 0:
            # Fallback: single-edge paths (always valid)
            result.append([[list(e)] for e in _instance(n, p, seed)])
            continue
        paths = _solve_instance(n, p, seed)
        edges = _instance(n, p, seed)
        if not _is_valid(paths, edges, n):
            paths = [[list(e)] for e in edges]
        result.append(paths)
    return result
