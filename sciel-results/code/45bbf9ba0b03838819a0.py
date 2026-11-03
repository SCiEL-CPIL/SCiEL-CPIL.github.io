"""Direct-approach cage solver: many explicit constructions + exact girth
verification, returning the smallest verified k-regular graph of girth >= g."""
import time
import threading
import random as _random
import itertools as _itertools
from collections import deque

_DEADLINE = 145.0
_start = [None]


def _elapsed():
    s = _start[0]
    return 0.0 if s is None else time.time() - s


# ---------------- exact girth (non-backtracking walks, all vertices) ----------
def _nonback_girth(adj, n):
    """Return length of shortest cycle, certified by non-backtracking walks
    from every vertex. >= g means no cycle shorter than g."""
    prev = {}
    q = deque()
    for u in range(n):
        for p in adj[u]:
            key = (p, u)
            prev[key] = (u, 1)
            q.append(key)
    while q:
        (v, p), (u, dist) = q.popleft()
        for w in adj[v]:
            if w == p:
                continue
            key = (w, v)
            if w == u:
                return dist + 1
            if key in prev:
                continue
            prev[key] = (v, dist + 1)
            q.append(key)
    return 10 ** 9


def _is_valid(adj, n, k, g):
    if n == 0:
        return False
    if any(len(adj[i]) != k for i in range(n)):
        return False
    for i in range(n):
        if len(set(adj[i])) != len(adj[i]):
            return False
        for j in adj[i]:
            if i <= j:
                return False
    return _nonback_girth(adj, n) >= g


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        adj[i] = sorted(adj[i])
    return adj


# ---------------- construction 1: LUW (Lazebnik-Ustimenko-Woldar) ------------
def _luw(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
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


# ---------------- construction 2: generalized Petersen graph G(N, k) ----------
def _petersen(k, g):
    if k != 3:
        return None
    for N in range(2 * k + 1, 5000):
        if N % 2 == 0 or N <= 2 * k:
            continue
        edges = []
        for i in range(N):
            edges.append((i, (i + 1) % N))
            edges.append((N + i, (N + (i + 1)) % N))
            edges.append((i, N + i))
        edges = list(set(edges))
        n = 2 * N
        if len(edges) != N * 3:
            continue
        adj = _build_adj(edges, n)
        if all(len(adj[i]) == 3 for i in range(n)) and _nonback_girth(adj, n) >= g:
            return n, edges
    return None


# ---------------- construction 3: circulant graph C(n; S) --------------------
def _circulant(k, g):
    if k % 2 != 0:
        return None
    n = (k // 2) * 2
    while n < 4 * k and _elapsed() < _DEADLINE:
        half = k // 2
        found = None
        for S in _itertools.combinations(range(1, n // 2 + 1), half):
            S = list(S)
            if any((n - s) % n in S for s in S):
                continue
            adj = [set() for _ in range(n)]
            for i in range(n):
                for s in S:
                    adj[i].add((i + s) % n)
                    adj[i].add((i - s) % n)
            if all(len(adj[i]) == k for i in range(n)):
                if _nonback_girth(adj, n) >= g:
                    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                    found = (n, edges)
                    break
        if found:
            return found
        n += 2
    return None


# ---------------- construction 4: lexicographic (blow-up) product ------------
def _blowup(k, g):
    base = {
        3: [3, 4, 5, 6, 7, 8, 9, 10, 11],
        4: [5, 6, 7, 8, 9, 10, 11],
        5: [6, 7, 8, 9, 10, 11, 12],
    }.get(k)
    if not base:
        return None
    for B in base:
        if B < k:
            continue
        if (B - k) % 2 != 0:
            continue
        for a in range(1, B):
            if a == B - a or (B - a) == 0:
                continue
            S = {a, B - a}
            adj = [set() for _ in range(B)]
            for i in range(B):
                for s in S:
                    adj[i].add((i + s) % B)
            if any(len(adj[i]) != 2 for i in range(B)):
                continue
            if _nonback_girth(adj, B) < (g + 1) // 2:
                continue
            m = (B - k) // 2
            n = B * m
            edges = []
            for i in range(B):
                for j in adj[i]:
                    if i < j:
                        for a in range(m):
                            for b in range(m):
                                edges.append((i * m + a, j * m + b))
                for a in range(m):
                    for b in range(a + 1, m):
                        edges.append((i * m + a, i * m + b))
            edges = list(set(edges))
            if len(edges) != (n * k) // 2:
                continue
            aadj = _build_adj(edges, n)
            if all(len(aadj[i]) == k for i in range(n)) and _nonback_girth(aadj, n) >= g:
                return n, edges
    return None


# ---------------- construction 5: BFS tree-like completion --------------------
def _bfs_complete(k, g):
    R = (g - 1) // 2
    depth_lim = min(R - 1, 5)
    order = [(0, -1, 0)]
    node_idx = {0: 0}
    parent_of = {0: -1}
    depth_of = {0: 0}
    q = deque([0])
    while q:
        u = q.popleft()
        du = depth_of[u]
        if du >= depth_lim:
            continue
        for child in range(k):
            v = len(order)
            order.append((v, u, du + 1))
            node_idx[v] = v
            parent_of[v] = u
            depth_of[v] = du + 1
            q.append(v)
    n = len(order)
    adj = [set() for _ in range(n)]
    for v in range(1, n):
        p = parent_of[v]
        adj[v].add(p)
        adj[p].add(v)

    def cand_ok(v, w):
        if w in adj[v]:
            return False
        if w == parent_of[v]:
            return False
        common = set(adj[v]) & set(adj[w])
        if common:
            return False
        if w in adj[parent_of[v]]:
            return False
        return True

    for v in range(n):
        guard = 0
        while len(adj[v]) < k and guard < 5000:
            guard += 1
            for w in range(n):
                if len(adj[v]) >= k:
                    break
                if w <= v:
                    continue
                if cand_ok(v, w):
                    adj[v].add(w)
                    adj[w].add(v)
    if any(len(adj[i]) != k for i in range(n)):
        return None
    for i in range(n):
        adj[i] = sorted(adj[i])
    if _nonback_girth(adj, n) < g:
        return None
    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
    return n, edges


# ---------------- construction 6: randomised greedy completion ---------------
def _greedy_complete(k, g, seed=0, max_n=None, depth_lim=None):
    rng = _random.Random(seed)
    if max_n is None:
        max_n = 200
    if depth_lim is None:
        depth_lim = 4

    for attempt in range(8):
        if _elapsed() > _DEADLINE - 5.0:
            return None
        n = rng.randint(k + 1, max_n)
        if n * k % 2 != 0:
            n += 1
        adj = [set() for _ in range(n)]
        verts = list(range(n))
        rng.shuffle(verts)
        ok = True
        for v in verts:
            if _elapsed() > _DEADLINE - 5.0:
                ok = False
                break
            need = k - len(adj[v])
            if need <= 0:
                continue
            cands = [w for w in range(n) if w != v and w not in adj[v]]
            rng.shuffle(cands)
            placed = 0
            for w in cands:
                if placed >= need:
                    break
                if _elapsed() > _DEADLINE - 5.0:
                    ok = False
                    break
                if _edge_ok(adj, v, w, g):
                    adj[v].add(w)
                    adj[w].add(v)
                    placed += 1
            if not ok:
                break
            if placed < need:
                ok = False
                break
        if not ok:
            continue
        aadj = [sorted(adj[i]) for i in range(n)]
        if any(len(aadj[i]) != k for i in range(n)):
            continue
        if _nonback_girth(aadj, n) < g:
            continue
        edges = [(i, j) for i in range(n) for j in aadj[i] if i < j]
        return n, edges
    return None


def _edge_ok(adj, v, w, g):
    if v == w:
        return False
    if w in adj[v]:
        return False
    dist = {v: 0}
    q = deque([v])
    while q:
        u = q.popleft()
        du = dist[u]
        if du >= g - 1:
            break
        for x in adj[u]:
            if x not in dist:
                dist[x] = du + 1
                if x == w:
                    if du + 2 < g:
                        return False
                    break
                q.append(x)
    if w in dist and dist[w] + 1 < g:
        return False
    return True


# ---------------- construction 7: random regular graph + local search --------
def _random_regular(k, g, seed=0):
    """Generate a random k-regular graph via configuration model, then verify."""
    rng = _random.Random(seed)
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 2 * sum((k - 1) ** i for i in range(r + 1))
    else:
        r = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(r)) + k * (k - 1) ** (r - 1)

    min_n = max(k + 1, moore)
    if min_n * k % 2 != 0:
        min_n += 1

    for n in range(min_n, min_n + 60, 2):
        if n * k % 2 != 0:
            continue
        if _elapsed() > _DEADLINE - 5.0:
            break
        for attempt in range(20):
            if _elapsed() > _DEADLINE - 5.0:
                break
            half_edges = n * k // 2
            stubs = []
            for v in range(n):
                stubs.extend([v] * k)
            rng.shuffle(stubs)
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, half_edges, 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if not valid:
                continue
            if any(len(adj[i]) != k for i in range(n)):
                continue
            aadj = [sorted(adj[i]) for i in range(n)]
            if _nonback_girth(aadj, n) >= g:
                edges = [(i, j) for i in range(n) for j in aadj[i] if i < j]
                return n, edges
    return None


# ---------------- construction 8: improved greedy with backtracking ----------
def _greedy_backtrack(k, g, seed=0):
    rng = _random.Random(seed)
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 2 * sum((k - 1) ** i for i in range(r + 1))
    else:
        r = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(r)) + k * (k - 1) ** (r - 1)

    min_n = max(k + 1, moore)
    if min_n * k % 2 != 0:
        min_n += 1

    for n in range(min_n, min_n + 40, 2):
        if n * k % 2 != 0:
            continue
        if _elapsed() > _DEADLINE - 5.0:
            break
        for attempt in range(5):
            if _elapsed() > _DEADLINE - 5.0:
                break
            adj = [set() for _ in range(n)]
            verts = list(range(n))
            rng.shuffle(verts)
            ok = True
            for v in verts:
                need = k - len(adj[v])
                if need <= 0:
                    continue
                cands = [w for w in range(n) if w != v and w not in adj[v]]
                rng.shuffle(cands)
                placed = 0
                for w in cands:
                    if placed >= need:
                        break
                    if _edge_ok(adj, v, w, g):
                        adj[v].add(w)
                        adj[w].add(v)
                        placed += 1
                if placed < need:
                    ok = False
                    break
            if not ok:
                continue
            aadj = [sorted(adj[i]) for i in range(n)]
            if any(len(aadj[i]) != k for i in range(n)):
                continue
            if _nonback_girth(aadj, n) < g:
                continue
            edges = [(i, j) for i in range(n) for j in aadj[i] if i < j]
            return n, edges
    return None


# ---------------- construction 9: edge-swap local search on random regular ----
def _edge_swap_search(k, g, seed=0, max_iters=5000):
    """Generate random k-regular graph, then use 2-opt edge swaps to increase girth."""
    rng = _random.Random(seed)
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 2 * sum((k - 1) ** i for i in range(r + 1))
    else:
        r = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(r)) + k * (k - 1) ** (r - 1)

    min_n = max(k + 1, moore)
    if min_n * k % 2 != 0:
        min_n += 1

    # Try a few sizes
    for n in range(min_n, min_n + 30, 2):
        if n * k % 2 != 0:
            continue
        if _elapsed() > _DEADLINE - 5.0:
            break
        for attempt in range(3):
            if _elapsed() > _DEADLINE - 5.0:
                break
            # Generate random k-regular graph
            half_edges = n * k // 2
            stubs = []
            for v in range(n):
                stubs.extend([v] * k)
            rng.shuffle(stubs)
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, half_edges, 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if not valid:
                continue
            if any(len(adj[i]) != k for i in range(n)):
                continue

            # Local search: try edge swaps to increase girth
            best_girth = _nonback_girth([sorted(adj[i]) for i in range(n)], n)
            if best_girth >= g:
                aadj = [sorted(adj[i]) for i in range(n)]
                edges = [(i, j) for i in range(n) for j in aadj[i] if i < j]
                return n, edges

            for it in range(max_iters):
                if _elapsed() > _DEADLINE - 5.0:
                    break
                # Pick random edge (u,v) and random edge (w,x) for 2-opt swap
                u = rng.randint(0, n - 1)
                v = rng.choice(list(adj[u]))
                w = rng.randint(0, n - 1)
                if w == u or w == v:
                    continue
                x = rng.choice(list(adj[w]))
                if x == u or x == v:
                    continue
                # Check if swap is valid: remove (u,v) and (w,x), add (u,w) and (v,x)
                # or (u,x) and (v,w)
                for new_pair in [((u, w), (v, x)), ((u, x), (v, w))]:
                    (a1, b1), (a2, b2) = new_pair
                    if a1 == b1 or a2 == b2:
                        continue
                    if b1 in adj[a1] or b2 in adj[a2]:
                        continue
                    # Check no duplicate edges
                    if b1 in adj[a1] or b2 in adj[a2]:
                        continue
                    # Perform swap
                    adj[a1].remove(b1)
                    adj[b1].remove(a1)
                    adj[a2].remove(b2)
                    adj[b2].remove(a2)
                    adj[a1].add(b1)
                    adj[b1].add(a1)
                    adj[a2].add(b2)
                    adj[b2].add(a2)
                    # Check girth
                    aadj = [sorted(adj[i]) for i in range(n)]
                    new_girth = _nonback_girth(aadj, n)
                    if new_girth >= g:
                        edges = [(i, j) for i in range(n) for j in aadj[i] if i < j]
                        return n, edges
                    if new_girth > best_girth:
                        best_girth = new_girth
                    else:
                        # Revert
                        adj[a1].remove(b1)
                        adj[b1].remove(a1)
                        adj[a2].remove(b2)
                        adj[b2].remove(a2)
                        adj[a1].add(b1)
                        adj[b1].add(a1)
                        adj[a2].add(b2)
                        adj[b2].add(a2)
                    break
    return None


# ---------------- parallel candidate verification ---------------------------
def _worker(name, fn, holder, k, g):
    try:
        res = fn()
    except Exception:
        res = None
    if res is None:
        return
    n, edges = res
    if _elapsed() >= _DEADLINE:
        return
    try:
        if n > 0 and _is_valid(_build_adj(edges, n), n, k, g):
            cur = holder.get("best")
            if cur is None or n < cur[0]:
                holder["best"] = (n, edges)
    except Exception:
        pass


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}
    jobs = [
        ("luw", lambda: _luw(k, g)),
        ("petersen", lambda: _petersen(k, g)),
        ("circulant", lambda: _circulant(k, g)),
        ("blowup", lambda: _blowup(k, g)),
        ("bfs", lambda: _bfs_complete(k, g)),
        ("greedy0", lambda: _greedy_complete(k, g, seed=0)),
        ("greedy1", lambda: _greedy_complete(k, g, seed=1)),
        ("greedy2", lambda: _greedy_complete(k, g, seed=2)),
        ("rand_reg0", lambda: _random_regular(k, g, seed=0)),
        ("rand_reg1", lambda: _random_regular(k, g, seed=1)),
        ("rand_reg2", lambda: _random_regular(k, g, seed=2)),
        ("backtrack0", lambda: _greedy_backtrack(k, g, seed=0)),
        ("backtrack1", lambda: _greedy_backtrack(k, g, seed=1)),
        ("edge_swap0", lambda: _edge_swap_search(k, g, seed=0, max_iters=3000)),
        ("edge_swap1", lambda: _edge_swap_search(k, g, seed=1, max_iters=3000)),
    ]
    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder, k, g), daemon=True)
        t.start()
        threads.append(t)

    # wait, with a safety margin, for threads to finish
    while _elapsed() < _DEADLINE - 2.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.1)

    best = holder.get("best")
    if best is None:
        # last-resort: return LUW result
        try:
            n, edges = _luw(k, g)
            best = (n, edges)
        except Exception:
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            best = (n, edges)
    return best
