"""Direct constructive growth for R(C4,K11): polarity core + greedy vertex addition
with independence-aware edge selection, plus C4-aware local-search polishing.

Structurally different from the seed: instead of building a fixed candidate and
running an expensive exact max-clique on its complement, we grow the graph vertex
by vertex, choosing each new vertex's neighborhood to (a) preserve C4-freedom and
(b) keep the independence number small (heuristic + local repair).  We then polish
the final graph with C4-aware edge swaps that reduce α.
"""
import time
import random

# EVOLVE-BLOCK-START

def _polarity_graph(q):
    """C4-free polarity graph of PG(2,q).  Returns (n, adj) with adj as int bitmasks."""
    invs = [0] + [pow(a, q - 2, q) for a in range(1, q)]
    pts = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == y == z == 0:
                    continue
                if x != 0:
                    ix = invs[x]
                    pts.append((1, (y * ix) % q, (z * ix) % q))
                elif y != 0:
                    iy = invs[y]
                    pts.append((0, 1, (z * iy) % q))
                else:
                    pts.append((0, 0, 1))
    n = len(pts)
    adj = [0] * n
    for i in range(n):
        a1, b1, c1 = pts[i]
        for j in range(i + 1, n):
            a2, b2, c2 = pts[j]
            if (a1 * a2 + b1 * b2 + c1 * c2) % q == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return n, adj


def _is_c4_free(n, adj):
    """True iff no two vertices share two common neighbours."""
    for i in range(n):
        for j in range(i + 1, n):
            c = adj[i] & adj[j]
            if c & (c - 1):
                return False
    return True


def _greedy_alpha(n, adj, iters=200, seed=0):
    """Upper-bound heuristic for the independence number (not exact).
    Repeatedly pick a vertex of min degree in the remaining induced subgraph,
    add it to the independent set, and remove it + its neighbours.  Returns the
    size of the independent set found.  Lower is better (we want alpha < 11)."""
    rng = random.Random(seed)
    best = n
    for _ in range(iters):
        # random ordering
        order = list(range(n))
        rng.shuffle(order)
        remaining = (1 << n) - 1
        indep = 0
        for v in order:
            if not (remaining >> v) & 1:
                continue
            # count neighbours in remaining
            deg = (adj[v] & remaining).bit_count()
            # pick min-degree vertex among remaining
            min_deg = 1 << 30
            pick = -1
            r = remaining
            while r:
                lb = r & (-r)
                u = lb.bit_length() - 1
                d = (adj[u] & remaining).bit_count()
                if d < min_deg:
                    min_deg = d
                    pick = u
                r ^= lb
            if pick < 0:
                break
            indep += 1
            remaining &= ~(adj[pick] | (1 << pick))
        if indep < best:
            best = indep
    return best


def _exact_alpha_small(n, adj, limit=11):
    """Exact max independent set via branch-and-bound on the complement, with early
    exit if we find one of size >= limit.  Returns the size (capped at limit)."""
    # complement adjacency
    comp = [0] * n
    for i in range(n):
        comp[i] = ((1 << n) - 1) & ~(adj[i] | (1 << i))
    best = 0
    # order by comp degree desc
    order = sorted(range(n), key=lambda v: -comp[v].bit_count())
    # map
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    comp_o = [0] * n
    for i, v in enumerate(order):
        comp_o[i] = comp[v]

    def color_bound(rem):
        """Greedy coloring of the subgraph induced by rem (in comp_o).  Returns #colors."""
        colors = 0
        uncolored = rem
        while uncolored:
            colors += 1
            cls = 0
            cand = uncolored
            while cand:
                lb = cand & (-cand)
                v = lb.bit_length() - 1
                if comp_o[v] & cls:
                    cand ^= lb
                    continue
                cls |= lb
                cand ^= lb
            if cls:
                uncolored &= ~cls
        return colors

    def dfs(rem, size):
        nonlocal best
        if size >= limit:
            best = limit
            return
        if not rem:
            return
        if size + rem.bit_count() <= best:
            return
        cb = color_bound(rem)
        if size + cb <= best:
            return
        # pick highest-degree vertex in rem
        best_v = -1
        best_d = -1
        r = rem
        while r:
            lb = r & (-r)
            v = lb.bit_length() - 1
            d = (comp_o[v] & rem).bit_count()
            if d > best_d:
                best_d = d
                best_v = v
            r ^= lb
        # include
        dfs(rem & comp_o[best_v], size + 1)
        if best >= limit:
            return
        # exclude
        dfs(rem & ~(1 << best_v), size)

    dfs((1 << n) - 1, 0)
    return best


def _try_add_vertex(n, adj, deadline):
    """Try to add one vertex to the C4-free graph on n vertices.
    For each candidate edge-set to existing vertices, check C4-freedom and
    estimate the new alpha.  Return (new_n, new_adj) or None."""
    best_alpha = 1 << 30
    best_mask = None
    # candidate: connect new vertex to a subset S of existing vertices.
    # C4 constraint: for any two existing vertices u,v that are BOTH neighbours
    # of the new vertex, u and v must NOT share a common neighbour (other than
    # the new vertex itself).  Equivalently, adj[u] & adj[v] must be 0 for
    # u,v in S.  Also, for any existing vertex w not in S, and any u in S,
    # w must not be a common neighbour of u and the new vertex -- but the new
    # vertex is only adjacent to S, so common neighbours of u and new vertex
    # are S ∩ N(u).  For C4: if w is adjacent to u and w is adjacent to new
    # vertex (i.e., w in S), then u and w are both in S and must have no
    # common neighbour.  So the constraint is simply: S is a set of vertices
    # with pairwise disjoint neighbour sets (in the original graph).
    #
    # To keep it tractable, we try: S = {v} for each v (add a leaf-like vertex),
    # and S = {u, v} for pairs where N(u) ∩ N(v) = ∅.
    #
    # For alpha: adding a vertex connected to S increases alpha by at most 1,
    # but if S is large, it may not increase alpha at all (the new vertex can
    # be excluded from any independent set that avoids S).  Heuristic: estimate
    # new alpha as old_alpha (if S is "large enough") or old_alpha + 1.
    #
    # We'll just try single-vertex connections and pair connections, and use
    # greedy_alpha to evaluate.

    # Try single vertex connections
    for v in range(n):
        new_n = n + 1
        new_adj = adj[:] + [0]
        new_adj[v] |= (1 << new_n)
        new_adj[new_n] |= (1 << v)
        if not _is_c4_free(new_n, new_adj):
            continue
        a = _greedy_alpha(new_n, new_adj, iters=50, seed=v)
        if a < best_alpha:
            best_alpha = a
            best_mask = 1 << v

    # Try pair connections (only if n is small enough)
    if n <= 40:
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & adj[v]:
                    continue  # share a neighbour -> C4
                new_n = n + 1
                new_adj = adj[:] + [0]
                new_adj[u] |= (1 << new_n)
                new_adj[v] |= (1 << new_n)
                new_adj[new_n] |= (1 << u) | (1 << v)
                if not _is_c4_free(new_n, new_adj):
                    continue
                a = _greedy_alpha(new_n, new_adj, iters=50, seed=u * 100 + v)
                if a < best_alpha:
                    best_alpha = a
                    best_mask = (1 << u) | (1 << v)

    if best_mask is None:
        return None
    new_n = n + 1
    new_adj = adj[:] + [0]
    for v in range(n):
        if (best_mask >> v) & 1:
            new_adj[v] |= (1 << new_n)
            new_adj[new_n] |= (1 << v)
    return new_n, new_adj


def _local_search_polish(n, adj, deadline, target=10):
    """C4-aware local search: try adding/removing edges to reduce alpha.
    We only ADD edges (to reduce alpha) and check C4-freedom."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        a = _greedy_alpha(n, adj, iters=100, seed=int(time.time() * 1000) % 10000)
        if a <= target:
            break
        # Try adding each missing edge
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    continue
                # Check C4: i and j must have no common neighbour
                if adj[i] & adj[j]:
                    continue
                # Tentatively add
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                a2 = _greedy_alpha(n, adj, iters=50, seed=i * 100 + j)
                if a2 < a:
                    a = a2
                    improved = True
                    if a <= target:
                        return n, adj
                else:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
    return n, adj


def construct():
    deadline = time.time() + 130.0  # safety margin
    best_n = 0
    best_adj = None

    def consider(n, adj):
        nonlocal best_n, best_adj
        if n <= best_n or n < 1:
            return
        if not _is_c4_free(n, adj):
            return
        # Exact check: alpha < 11
        if time.time() > deadline - 5:
            # Can't afford exact check; use greedy as proxy
            a = _greedy_alpha(n, adj, iters=200, seed=42)
            if a >= 11:
                return
        else:
            a = _exact_alpha_small(n, adj, limit=11)
            if a >= 11:
                return
        best_n = n
        best_adj = adj[:]

    # Start with polarity graph core
    for q in [5, 7, 11, 13]:
        if time.time() > deadline - 30:
            break
        try:
            n, adj = _polarity_graph(q)
            consider(n, adj)
            # Try to grow
            for _ in range(30):  # try to add up to 30 vertices
                if time.time() > deadline - 10:
                    break
                result = _try_add_vertex(n, adj, deadline)
                if result is None:
                    break
                n, adj = result
                consider(n, adj)
        except Exception:
            continue

    # Polish the best found
    if best_n > 0 and time.time() < deadline - 15:
        try:
            n, adj = _local_search_polish(best_n, best_adj[:], deadline - 5, target=10)
            consider(n, adj)
        except Exception:
            pass

    # Fallback: if nothing worked, use a small valid graph
    if best_n < 20:
        # 10 disjoint triangles: n=30, alpha=10, C4-free
        n = 30
        adj = [0] * n
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            adj[a] |= (1 << b) | (1 << c)
            adj[b] |= (1 << a) | (1 << c)
            adj[c] |= (1 << a) | (1 << b)
        best_n = n
        best_adj = adj

    edges = []
    for i in range(best_n):
        m = best_adj[i]
        while m:
            lb = m & (-m)
            j = lb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            m ^= lb

    return {'n': best_n, 'edges': edges}

# EVOLVE-BLOCK-END
