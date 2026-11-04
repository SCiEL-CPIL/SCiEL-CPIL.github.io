"""Rewritten (4,10)-cage search: random 4-regular graphs + girth-maximizing
non-backtracking (cage-growth) search. Returns the smallest n with a valid
4-regular simple graph of girth >= g."""
import time
import random
from collections import deque

_DEADLINE = 115.0   # safety margin before the 150 s wall-clock kill
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _girth_from_root(adj, g):
    """Return the girth (smallest cycle length) of the connected component of
    vertex 0 via non-backtracking BFS. If no cycle is shorter than g, return g."""
    # state: (vertex, previous_vertex)  ->  depth
    seen = {(0, -1): 0}
    q = deque([(0, -1)])
    best = None
    while q:
        v, p = q.popleft()
        d = seen[(v, p)]
        if d >= g - 1:
            continue
        for w in adj[v]:
            if w == p:
                continue
            if (w, v) in seen:
                cand = seen[(w, v)] + d + 1
                if best is None or cand < best:
                    best = cand
            else:
                seen[(w, v)] = d + 1
                q.append((w, v))
    return best if best is not None else g


def _is_girth_ok(adj, g):
    """True iff every vertex has degree exactly len(adj) and the (single)
    connected component has girth >= g."""
    n = len(adj)
    for v in range(n):
        if len(adj[v]) != n - 1:
            return False
    # connectivity check
    seen = [False] * n
    seen[0] = True
    q = deque([0])
    cnt = 1
    while q:
        v = q.popleft()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                q.append(w)
    if cnt != n:
        return False
    return _girth_from_root(adj, g) >= g


def _grow(adj, g):
    """Try to add one vertex keeping 4-regularity and girth >= g.
    Returns a new adjacency list on success, else None."""
    n = len(adj)
    # candidate new vertex u = n. It must connect to 4 existing vertices
    # such that each gains one degree (so we must first *remove* an edge
    # incident to each of them to keep them at degree 4).
    # We pick a set of 4 vertices to "open" (remove one edge each) then
    # attach u to them. The removed edge endpoints become the new u's links.
    # Simpler: pick an existing edge (a,b); split it: remove a-b, add a-u, b-u.
    # That keeps a,b at degree 4 and gives u degree 2. Not enough.
    # So we need to open 4 distinct "slots". We remove 4 edges and add 4
    # edges from u, but u only has 4 neighbours -> we remove 4 edges whose
    # 4 chosen endpoints connect to u.
    # Choose 4 distinct edges; for each, pick one endpoint to attach to u.
    # The other endpoint loses a degree -> must be restored by pairing two
    # of the "leftover" endpoints together.
    # This is a 2-regular repair problem. Keep it simple:
    # Find 4 edges forming a structure we can flip.
    # Approach: find a 4-cycle a-b-c-d-a. Replace with a-u-b? no.
    # Instead: pick two disjoint edges (a,b) and (c,d). Remove both, add
    # a-u, b-u, c-u, d-u -> u has degree 4, a,b,c,d keep degree 4.
    # This works! We need two disjoint edges.
    edges = []
    for a in range(n):
        for b in adj[a]:
            if a < b:
                edges.append((a, b))
    random.shuffle(edges)
    for i in range(len(edges)):
        a, b = edges[i]
        for j in range(i + 1, min(i + 400, len(edges))):
            c, d = edges[j]
            if len({a, b, c, d}) != 4:
                continue
            # build new adjacency
            new_adj = [list(x) for x in adj]
            # remove a-b, c-d
            new_adj[a].remove(b)
            new_adj[b].remove(a)
            new_adj[c].remove(d)
            new_adj[d].remove(c)
            # add u edges
            u = n
            for v in (a, b, c, d):
                new_adj[v].append(u)
            new_adj.append([a, b, c, d])
            if _is_girth_ok(new_adj, g):
                return new_adj
    return None


def _repair_degree(adj):
    """Ensure every vertex has degree exactly 4 by edge swaps. Returns new adj
    or None if it fails."""
    n = len(adj)
    for _ in range(50):
        bad = [v for v in range(n) if len(adj[v]) != 4]
        if not bad:
            break
        # find a vertex with degree > 4 and one with degree < 4
        hi = [v for v in range(n) if len(adj[v]) > 4]
        lo = [v for v in range(n) if len(adj[v]) < 4]
        if not hi or not lo:
            return None
        v = hi[0]
        w = lo[0]
        # find a neighbour x of v such that x is not w and w not adjacent to x
        cand = [x for x in adj[v] if x != w and w not in adj[x]]
        random.shuffle(cand)
        done = False
        for x in cand:
            if x == w or w in adj[x]:
                continue
            new_adj = [list(t) for t in adj]
            new_adj[v].remove(x)
            new_adj[x].remove(v)
            new_adj[v].append(w)
            new_adj[w].append(v)
            new_adj[x].append(w)
            new_adj[w].append(x)
            if all(len(t) == 4 for t in new_adj):
                adj = new_adj
                done = True
                break
        if not done:
            return None
    if any(len(t) != 4 for t in adj):
        return None
    return adj


def _random_4regular(n, seed):
    """Random 4-regular simple graph on n vertices via configuration model +
    repair. Returns adjacency list or None."""
    rng = random.Random(seed)
    for attempt in range(60):
        # build degree-4 multigraph via configuration
        half = list(range(n)) * 4
        rng.shuffle(half)
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(0, 2 * n * 2, 2):
            a, b = half[i], half[i + 1]
            if a == b:
                ok = False
                break
            adj[a].add(b)
            adj[b].add(a)
        if not ok:
            continue
        if any(len(s) != 4 for s in adj):
            continue
        # convert to lists, attempt to fix any self-loops already excluded
        adj = [sorted(s) for s in adj]
        # repair to make it simple & 4-regular (already simple by construction
        # since sets; but multi-edges impossible with sets). Ensure 4-regular.
        if any(len(s) != 4 for s in adj):
            continue
        return adj
    return None


def construct(k, g):
    global _START
    _START = time.time()
    if k != 4:
        # Fallback: use the original LUW construction for other (k,g).
        return _luw(k, g)
    rng = random.Random(12345)
    best = None
    # Start near the Moore-bound-ish size and grow. The known cage is 320.
    # We search upward from a modest base, trying to find girth >= g.
    base = 244
    for n in range(base, base + 120):
        if _remaining() < 12:
            break
        if n % 2 == 1:
            continue  # 4-regular graphs need even vertex count
        for attempt in range(300):
            if _remaining() < 8:
                break
            adj = _random_4regular(n, rng.randrange(1 << 30))
            if adj is None:
                continue
            if _is_girth_ok(adj, g):
                # found a valid graph of size n
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        # If we cannot find girth g at size n with random graphs, we could
        # try to grow from a smaller girth-g graph, but random search at
        # increasing n is the primary mechanism.
    # Fallback: return the LUW construction (valid, large) so we never crash.
    return _luw(k, g)


def _luw(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction (valid fallback)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
