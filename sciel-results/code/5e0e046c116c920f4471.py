"""GP-graph (voltage-lift) search for the (3,16) cage, with LUW fallback."""
import time
import random
import numpy as np

_DEADLINE = 150.0
_EPS = 5.0


def construct(k, g):
    t0 = time.time()

    def deadline():
        return (time.time() - t0) > (_DEADLINE - _EPS)

    if k != 3:
        return _luw(k, g)

    host, edges, n = _make_host()
    nL, nR = n // 2, n // 2
    degL, degR = 3, 3
    degs = degL * nL + degR * nR

    # ---- build non-backtracking closed walks of length < g (certification set) ----
    closed = _build_closed_walks(host, g, deadline)
    if closed is None:
        return _luw(k, g)

    # ---- search voltage assignments (balanced + girth-closing avoidance) ----
    result = _search_gp(host, edges, nL, nR, degL, degR, degs, closed,
                        g, deadline)
    if result is not None:
        return result

    # ---- fallback: always-valid algebraic construction ----
    return _luw(k, g)


# --------------------------------------------------------------------------- #
# Host graph (fixed 936-vertex bipartite cage-like structure)                 #
# --------------------------------------------------------------------------- #
def _make_host():
    nL = nR = 468
    n = nL + nR
    rng = random.Random(20260401)
    perm = list(range(nR)); rng.shuffle(perm)
    edges = []
    for i in range(nL):
        a = i
        b = (i * 7) % nR
        c = perm[i]
        if b == a or c == a or c == b:
            c = (a + 1 + (i % (nR - 2))) % nR
            if c == a or c == b:
                c = (a + 2 + (i % (nR - 3))) % nR
        edges.append((i, nR + a))
        edges.append((i, nR + b))
        edges.append((i, nR + c))
    host = [[] for _ in range(n)]
    for (u, v) in edges:
        host[u].append(v)
        host[v].append(u)
    return host, edges, n


def _build_closed_walks(host, g, deadline):
    """All non-backtracking closed walks of length 3..g-1 as edge-index lists."""
    n = len(host)
    m = len(host[0]) * n // 2  # not used; use len(host) adjacency
    eidx = {}
    for u in range(n):
        for v in host[u]:
            if u < v:
                eidx[(u, v)] = len(eidx)
    E = len(eidx)
    out = []
    for L in range(3, g):
        if deadline():
            return None
        # state: (prev, cur)
        for u in range(n):
            for v in host[u]:
                if u == v:
                    continue
                # walk u -> v, length 1
                if L == 1:
                    continue
                # extend
                # represent as list of vertices; track prev to forbid backtrack
                stack = [(v, u, [u, v])]
                while stack:
                    cur, prev, path = stack.pop()
                    if len(path) == L:
                        if path[0] == cur:
                            # closed; record edge indices
                            eids = []
                            ok = True
                            for a in range(len(path) - 1):
                                key = (path[a], path[a + 1])
                                if key in eidx:
                                    eids.append(eidx[key])
                                else:
                                    eids.append(eidx[(path[a + 1], path[a])])
                            out.append(tuple(eids))
                        continue
                    for w in host[cur]:
                        if w == prev:
                            continue
                        stack.append((w, cur, path + [w]))
    return out


# --------------------------------------------------------------------------- #
# GP search over edge-voltages in Z_m                                       #
# --------------------------------------------------------------------------- #
def _search_gp(host, edges, nL, nR, degL, degR, degs, closed, g, deadline):
    E = len(edges)
    # incidence: for each vertex, list of (edge_index, sign)  (+1 if vertex is
    # the smaller endpoint u, -1 otherwise) so that sum(sign*val[e]) = 0 (mod m)
    inc = [[] for _ in range(nL + nR)]
    for ei, (u, v) in enumerate(edges):
        inc[u].append((ei, 1))
        inc[v].append((ei, -1))

    # girth constraints: for each closed walk, sum over edges (with multiplicity)
    # must be != 0 (mod m).  Build (edge, count) per walk.
    con = []
    for walk in closed:
        cnt = {}
        for ei in walk:
            cnt[ei] = cnt.get(ei, 0) + 1
        con.append(list(cnt.items()))

    # try several moduli; small m is cheaper and often sufficient
    for m in (2, 3, 4, 5, 7):
        if deadline():
            break
        sol = _solve_csp(E, inc, con, m, deadline)
        if sol is not None:
            return _build_lift(edges, sol, nL, nR, degL, degR, degs)
    return None


def _solve_csp(E, inc, con, m, deadline):
    """Assign edge voltages in Z_m so every vertex is balanced and no closed
    walk is closed in the lift.  Backtracking with forward checking."""
    # order edges; precompute which girth constraints each edge participates in
    edge_con = [[] for _ in range(E)]
    for ci, items in enumerate(con):
        for ei, _ in items:
            edge_con[ei].append(ci)

    # choose variable order by degree in constraint graph (heuristic)
    order = sorted(range(E), key=lambda e: -len(edge_con[e]))

    val = [-1] * E
    # per-vertex current sum (mod m); per-constraint current sum (mod m)
    vsum = [0] * len(inc)
    csum = [0] * len(con)

    # constraint involvement per vertex for propagation
    # For backtracking we just recompute feasibility cheaply.

    # candidate values per edge
    # We use a simple recursive backtracking with the "distinct incident" rule
    # enforced at leaves and girth-closure checked incrementally.

    # To keep it fast, assign in order and prune:
    #  - after assigning an edge, update vsum of its two endpoints and csum of
    #    its constraints.
    #  - prune a vertex if its assigned incident edges already force two equal
    #    values (distinctness violated) OR if remaining capacity is impossible.
    #  - prune a constraint if it is already closed (csum == 0) before all its
    #    edges are assigned (then it can never become non-zero... actually it
    #    could change with remaining edges; so we only prune if all assigned).

    # Track assigned count per vertex and per constraint
    vcnt = [0] * len(inc)
    ccnt = [0] * len(con)
    total_vc = [len(x) for x in inc]
    total_cc = [len(x) for x in con]
    vvals = [set() for _ in range(len(inc))]

    # Precompute, for each edge, its two vertices
    everts = []
    for ei, (u, v) in enumerate(_edges_from_inc(inc)):
        everts.append((u, v))

    def feasible_vertex(vi):
        # distinctness: assigned values must be distinct
        # capacity: number of distinct remaining values >= remaining edges
        assigned = len(vvals[vi])
        remaining = total_vc[vi] - vcnt[vi]
        available = m - assigned
        return available >= remaining

    def feasible_con(ci):
        # if fully assigned, must be non-zero
        if ccnt[ci] == total_cc[ci]:
            return csum[ci] % m != 0
        # otherwise always tentatively feasible (could still reach non-zero)
        return True

    nodes = [0]

    def bt(pos):
        if deadline():
            return False
        if pos == E:
            # final validation
            for ci in range(len(con)):
                if csum[ci] % m == 0:
                    return False
            for vi in range(len(inc)):
                if vsum[vi] % m != 0:
                    return False
            return True
        ei = order[pos]
        u, v = everts[ei]
        # values to try: avoid equal to already-assigned incident values
        used = vvals[u] | vvals[v]
        for x in range(m):
            if x in used:
                continue
            # tentatively apply
            val[ei] = x
            vsum[u] = (vsum[u] + x) % m
            vsum[v] = (vsum[v] - x) % m
            vcnt[u] += 1; vcnt[v] += 1
            vvals[u].add(x); vvals[v].add(x)
            for ci in edge_con[ei]:
                csum[ci] = (csum[ci] + x) % m
                ccnt[ci] += 1
            # feasibility
            ok = feasible_vertex(u) and feasible_vertex(v)
            if ok:
                for ci in edge_con[ei]:
                    if not feasible_con(ci):
                        ok = False
                        break
            if ok:
                nodes[0] += 1
                if bt(pos + 1):
                    return True
            # undo
            for ci in edge_con[ei]:
                csum[ci] = (csum[ci] - x) % m
                ccnt[ci] -= 1
            vvals[u].discard(x); vvals[v].discard(x)
            vcnt[u] -= 1; vcnt[v] -= 1
            vsum[u] = (vsum[u] - x) % m
            vsum[v] = (vsum[v] + x) % m
            val[ei] = -1
        return False

    if bt(0):
        return val
    return None


def _edges_from_inc(inc):
    # reconstruct edges from incidence (u is +1 endpoint)
    n = len(inc)
    # map edge index -> (u,v); we don't have it here, so caller must pass edges.
    # This helper is a placeholder; real code passes edges directly.
    return []


def _build_lift(edges, sol, nL, nR, degL, degR, degs):
    n = nL + nR
    # voltage of an edge: +sol[e] from u to v, -sol[e] from v to u
    # vertex (v, t) connects to (w, t + voltage(v->w))
    # For u (left) -> v (right): voltage = +sol[ei]
    # For v (right) -> u (left): voltage = -sol[ei]
    m = None
    # infer modulus from sol values
    mv = max(sol) + 1
    # we don't know m here; pass it. Rebuild with explicit m.
    raise RuntimeError("use _build_lift_m")


# We need modulus in _build_lift; refactor _search_gp to return (sol, m).
def _search_gp(host, edges, nL, nR, degL, degR, degs, closed, g, deadline):
    E = len(edges)
    inc = [[] for _ in range(nL + nR)]
    for ei, (u, v) in enumerate(edges):
        inc[u].append((ei, 1))
        inc[v].append((ei, -1))
    con = []
    for walk in closed:
        cnt = {}
        for ei in walk:
            cnt[ei] = cnt.get(ei, 0) + 1
        con.append(list(cnt.items()))
    everts = [(u, v) for (u, v) in edges]

    for m in (2, 3, 4, 5, 7):
        if deadline():
            break
        sol = _solve_csp2(E, inc, con, everts, m, deadline)
        if sol is not None:
            return _build_lift_m(edges, sol, m, nL, nR)
    return None


def _solve_csp2(E, inc, con, everts, m, deadline):
    edge_con = [[] for _ in range(E)]
    for ci, items in enumerate(con):
        for ei, _ in items:
            edge_con[ei].append(ci)
    order = sorted(range(E), key=lambda e: -len(edge_con[e]))
    val = [-1] * E
    vsum = [0] * len(inc)
    csum = [0] * len(con)
    vcnt = [0] * len(inc)
    ccnt = [0] * len(con)
    total_vc = [len(x) for x in inc]
    total_cc = [len(x) for x in con]
    vvals = [set() for _ in range(len(inc))]

    def feasible_vertex(vi):
        return (m - len(vvals[vi])) >= (total_vc[vi] - vcnt[vi])

    def feasible_con(ci):
        if ccnt[ci] == total_cc[ci]:
            return csum[ci] % m != 0
        return True

    def bt(pos):
        if deadline():
            return False
        if pos == E:
            for ci in range(len(con)):
                if csum[ci] % m == 0:
                    return False
            for vi in range(len(inc)):
                if vsum[vi] % m != 0:
                    return False
            return True
        ei = order[pos]
        u, v = everts[ei]
        used = vvals[u] | vvals[v]
        for x in range(m):
            if x in used:
                continue
            val[ei] = x
            vsum[u] = (vsum[u] + x) % m
            vsum[v] = (vsum[v] - x) % m
            vcnt[u] += 1; vcnt[v] += 1
            vvals[u].add(x); vvals[v].add(x)
            for ci in edge_con[ei]:
                csum[ci] = (csum[ci] + x) % m
                ccnt[ci] += 1
            ok = feasible_vertex(u) and feasible_vertex(v)
            if ok:
                for ci in edge_con[ei]:
                    if not feasible_con(ci):
                        ok = False
                        break
            if ok and bt(pos + 1):
                return True
            for ci in edge_con[ei]:
                csum[ci] = (csum[ci] - x) % m
                ccnt[ci] -= 1
            vvals[u].discard(x); vvals[v].discard(x)
            vcnt[u] -= 1; vcnt[v] -= 1
            vsum[u] = (vsum[u] - x) % m
            vsum[v] = (vsum[v] + x) % m
            val[ei] = -1
        return False

    if bt(0):
        return val
    return None


def _build_lift_m(edges, sol, m, nL, nR):
    n = nL + nR
    edges_out = []
    for ei, (u, v) in enumerate(edges):
        x = sol[ei] % m
        for t in range(m):
            # u (left) -> v (right): v' = t + x
            a = u * m + t
            b = v * m + ((t + x) % m)
            edges_out.append((a, b))
    return m * n, edges_out


# --------------------------------------------------------------------------- #
# Fallback: Lazebnik-Ustimenko-Woldar construction (always valid)            #
# --------------------------------------------------------------------------- #
def _luw(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
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
