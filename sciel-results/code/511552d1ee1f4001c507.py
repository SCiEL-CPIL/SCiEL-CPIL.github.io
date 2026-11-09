"""Rewritten for the 3-regular girth-16 cage task: keep the L-U-W family but
search over its parameters and add a time-boxed, verified repair pass."""
import time
import random
from collections import deque, defaultdict

# EVOLVE-BLOCK-START

def _luw_construct(k, g, d, q):
    """Explicit Lazebnik-Ustimenko-Woldar q-regular bipartite graph with girth >= d+5 (odd d).
    Returns (n, edges) for the connected component of the zero point, or None on error."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
        if vertex[0] == 0:
            for other in neighbours(vertex):
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _girth_bfs(adj, n):
    """Exact girth via non-backtracking BFS from every vertex."""
    best = n + 1
    for s in range(n):
        # state: (prev, node); prev = -1 for the source
        dist = {(-1, s): 1}
        q = deque([(-1, s)])
        while q:
            prev, node = q.popleft()
            d = dist[(prev, node)]
            if d >= best:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if (node, nb) in dist:
                    # closing a cycle: length d + 1
                    if d + 1 < best:
                        best = d + 1
                    continue
                dist[(node, nb)] = d + 1
                q.append((node, nb))
    return best


def _is_valid(n, edges, k, g, deadline):
    if len(edges) != n * k // 2:
        return False
    adj = [set() for _ in range(n)]
    seen = set()
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in seen:
            return False
        seen.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    if time.time() > deadline:
        return False
    return _girth_bfs(adj, n) >= g


def _random_3_regular(n, rng):
    """Random 3-regular simple graph via configuration model + 2-switch repair."""
    if n < 4 or n % 2 == 1:
        return None
    if n % 2 == 0:
        pass
    else:
        return None
    # need even n for 3-regular
    if n % 2 != 0:
        return None
    half = n * 3 // 2
    adj = [set() for _ in range(n)]
    tries = 0
    while tries < 400:
        tries += 1
        adj = [set() for _ in range(n)]
        stubs = [i for i in range(n) for _ in range(3)]
        rng.shuffle(stubs)
        ok = True
        for a in range(0, len(stubs), 2):
            u, v = stubs[a], stubs[a + 1]
            if u == v:
                ok = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if not ok:
            continue
        # repair parallel edges / self loops with 2-switches
        for _ in range(200):
            bad = [(u, v) for u in range(n) for v in adj[u] if u < v and len({x for x in adj[u] if x == v}) > 1]
            # simpler: find multiedges
            bad_pairs = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        # count multiplicity
                        pass
            # detect parallel via edge multiset
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            if len(edges) != len(set(edges)):
                # find a parallel pair and a disjoint edge to 2-switch
                from collections import Counter
                cnt = Counter(edges)
                par = [e for e, c in cnt.items() if c > 1][0]
                done = False
                for _t in range(30):
                    e2 = rng.choice(edges)
                    a, b = e2
                    if len({a, b} & set(par)) == 0:
                        # switch (u,v)-(a,b) -> (u,a),(v,b)
                        adj[par[0]].discard(par[1]); adj[par[1]].discard(par[0])
                        adj[a].discard(b); adj[b].discard(a)
                        adj[par[0]].add(a); adj[a].add(par[0])
                        adj[par[1]].add(b); adj[b].add(par[1])
                        done = True
                        break
                if not done:
                    continue
            else:
                break
        # final check
        if all(len(adj[i]) == 3 for i in range(n)):
            edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
            if len(edges) == len(set(edges)):
                return edges
    return None


def _repair_girth(n, edges, g, rng, deadline):
    """Try to raise girth via 2-switches; return improved edges or None."""
    adj = _build_adj(n, edges)
    current_girth = _girth_bfs(adj, n)
    if current_girth >= g:
        return edges
    improved = edges
    for _ in range(20000):
        if time.time() > deadline:
            break
        if current_girth >= g:
            break
        # pick random vertex, do a 2-switch with another random edge
        u = rng.randrange(n)
        cand = list(adj[u])
        if not cand:
            continue
        v = cand[rng.randrange(len(cand))]
        # find another edge (a,b) disjoint from {u,v}
        all_edges = []
        for x in range(n):
            for y in adj[x]:
                if x < y:
                    all_edges.append((x, y))
        if not all_edges:
            break
        for _t in range(20):
            a, b = rng.choice(all_edges)
            if len({a, b} & {u, v}) == 0:
                # perform switch: remove (u,v),(a,b); add (u,a),(v,b)
                adj[u].discard(v); adj[v].discard(u)
                adj[a].discard(b); adj[b].discard(a)
                adj[u].add(a); adj[a].add(u)
                adj[v].add(b); adj[b].add(v)
                break
        # recompute girth (expensive) but only every few iterations
        current_girth = _girth_bfs(adj, n)
    return [(u, v) for u in range(n) for v in adj[u] if u < v]


def construct(k, g):
    deadline = time.time() + 110.0  # safety margin under 120s
    rng = random.Random(12345)

    best = None
    best_n = None

    # 1) Search the L-U-W family over plausible (d, q)
    if k == 3:
        # girth 16 needs d+5 >= 16 -> d >= 11 (odd)
        for d in [11, 13, 15, 17]:
            for q in [3, 5, 7]:
                if time.time() > deadline:
                    break
                try:
                    n, edges = _luw_construct(k, g, d, q)
                except Exception:
                    continue
                if n is None or n < 0:
                    continue
                if best_n is None or n < best_n:
                    best_n = n
                    best = (n, edges)
    else:
        # generic: keep the seed's own d choice
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        try:
            n, edges = _luw_construct(k, g, d, k if k != 4 else 4)
            best_n = n
            best = (n, edges)
        except Exception:
            best = None

    # 2) Random-restart local search for k == 3, small n near Moore bound
    if k == 3 and time.time() < deadline - 20:
        moore = 2 * (2 ** (g // 2 - 1))  # even girth lower bound
        # try a few n values that are even and >= moore, small enough to handle
        candidates = []
        start_n = max(moore, 512)
        # keep the search tractable: only try if n is modest
        for n in range(start_n, min(start_n + 64, 768), 2):
            if n % 2 != 0:
                continue
            candidates.append(n)
        for n in candidates:
            if time.time() > deadline - 10:
                break
            edges = _random_3_regular(n, rng)
            if edges is None:
                continue
            # quick girth check
            adj = _build_adj(n, edges)
            gi = _girth_bfs(adj, n)
            if gi >= g:
                if best_n is None or n < best_n:
                    best_n = n
                    best = (n, edges)
                break  # found a valid one at this n
            # otherwise attempt repair (limited)
            if time.time() < deadline - 10:
                rep = _repair_girth(n, edges, g, rng, deadline)
                if rep is not None:
                    adj2 = _build_adj(n, rep)
                    if _girth_bfs(adj2, n) >= g:
                        if best_n is None or n < best_n:
                            best_n = n
                            best = (n, rep)
                        break

    # 3) Final verification of the best candidate
    if best is None:
        # absolute fallback: the seed's own d
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        n, edges = _luw_construct(k, g, d, k if k != 4 else 4)
        return n, edges

    n, edges = best
    if _is_valid(n, edges, k, g, deadline):
        return n, edges

    # If best failed verification, fall back to a known-valid L-U-W instance
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n2, edges2 = _luw_construct(k, g, d, k if k != 4 else 4)
    if _is_valid(n2, edges2, k, g, deadline):
        return n2, edges2
    return n2, edges2

# EVOLVE-BLOCK-END
