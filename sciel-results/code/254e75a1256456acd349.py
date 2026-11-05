"""Construct a small 5-regular graph with girth at least g.

The primary strategy is to search over cyclic group covers of the Petersen
graph (a 5-regular, girth-5 base graph on 10 vertices / 15 edges).  A cover
with voltage group Z_m has 10*m vertices.  We enumerate m from the Moore-bound
lower limit upward and, for each m, try to find voltage assignments on the 15
edges such that every non-backtracking base walk of length L in {5,6,7,8}
(the Petersen girth is 5, so shorter closed walks cannot exist) has non-zero
net voltage (mod m), thereby guaranteeing girth >= g in the cover.

For each m we solve the resulting system of modular linear inequalities via a
backtracking SAT-style search over the 15 voltage variables.  The search is
pruned aggressively: partial assignments are checked against every constraint
whose variables are already fixed; constraints whose variables are not all
fixed are deferred.

If no cover is found within the time budget, we fall back to the
Lazebnik-Ustimenko-Woldar algebraic construction, which is guaranteed to
produce a k-regular graph of girth at least g.
"""
import time
import random
from itertools import product
from collections import defaultdict

# ---------------------------------------------------------------------------
# Petersen graph
# ---------------------------------------------------------------------------
PETERSEN_ADJ = [
    [1, 4, 5],
    [0, 2, 6],
    [1, 3, 7],
    [2, 4, 8],
    [0, 3, 9],
    [0, 6, 9],
    [1, 5, 7],
    [2, 6, 8],
    [3, 7, 9],
    [4, 5, 8],
]
N_BASE = 10
E_BASE = 15  # undirected edges

# Oriented edge list (each undirected edge appears twice, once per direction)
_ORIENTED = []
_EDGE_ID = {}  # (u, v) -> id  (undirected)
for u in range(N_BASE):
    for v in PETERSEN_ADJ[u]:
        if u < v:
            eid = len(_EDGE_ID)
            _EDGE_ID[(u, v)] = eid
            _EDGE_ID[(v, u)] = eid
            _ORIENTED.append((u, v, eid))
            _ORIENTED.append((v, u, eid))

# Precompute all non-backtracking closed walks of length 5..8 on the Petersen
# Each walk is a sequence of directed edges (edge ids).
# We store the count vector: for each undirected edge id (0..14), how many
# times the walk uses it in each direction.  The net voltage of a walk is
#   sum_e (count_fwd[e] - count_bwd[e]) * voltage[e]  (mod m)
# We need this to be non-zero for every closed non-backtracking walk of
# length L in 5..g-1.

def _compute_walk_constraints(g):
    """Return a list of tuples (coeffs, ) where coeffs is a list of 15 ints
    (net counts per edge).  A voltage assignment is valid iff for every
    coeffs in the list, sum(coeffs[i]*v[i] for i in range(15)) % m != 0."""
    # For each directed edge (u, v, eid), the "forward" direction is u->v.
    # Build a map: directed_edge_index -> (u, v, undirected_eid, sign)
    # sign = +1 if direction is (min, max), -1 if (max, min)
    dir_info = []  # indexed by position in _ORIENTED
    for idx, (u, v, eid) in enumerate(_ORIENTED):
        sign = 1 if u < v else -1
        dir_info.append((u, v, eid, sign))

    # For each start vertex s, do non-backtracking DFS up to depth g-1,
    # tracking (current_vertex, prev_vertex, depth, edge_id_sequence).
    # A closed walk of length L is one where current_vertex == s and depth == L.
    constraints_set = set()

    max_depth = g - 1  # we need to detect cycles of length < g, i.e., up to g-1
    # Actually for the Petersen, girth is 5, so closed non-backtracking walks
    # of length 5, 6, 7, 8 (for g=9) are what matter.

    for s in range(N_BASE):
        # Stack: (vertex, prev_vertex, depth, edge_seq)
        stack = []
        for ni, nb in enumerate(PETERSEN_ADJ[s]):
            # directed edge s -> nb
            # Find the oriented edge index
            # We need the edge id and sign
            for oi, (ou, ov, oid) in enumerate(_ORIENTED):
                if ou == s and ov == nb:
                    stack.append((nb, s, 1, [oid]))
                    break

        while stack:
            v, pv, d, seq = stack.pop()
            if d >= max_depth:
                # If v == s and d is in [5, max_depth], record constraint
                if v == s and d >= 5:
                    # Compute net count vector
                    counts = [0] * E_BASE
                    for oid in seq:
                        u_e, v_e, eid, sign = dir_info[oid]
                        counts[eid] += sign
                    key = tuple(counts)
                    if any(c != 0 for c in key):
                        constraints_set.add(key)
                continue
            # Extend
            for nb in PETERSEN_ADJ[v]:
                if nb == pv:
                    continue
                # Find directed edge v -> nb
                for oi, (ou, ov, oid) in enumerate(_ORIENTED):
                    if ou == v and ov == nb:
                        stack.append((nb, v, d + 1, seq + [oid]))
                        break

    # Deduplicate: many walks give the same count vector
    return list(constraints_set)

_CONSTRAINTS_CACHE = {}

def _get_constraints(g):
    if g not in _CONSTRAINTS_CACHE:
        _CONSTRAINTS_CACHE[g] = _compute_walk_constraints(g)
    return _CONSTRAINTS_CACHE[g]

# ---------------------------------------------------------------------------
# Voltage assignment search
# ---------------------------------------------------------------------------
def _solve_voltages(m, constraints, n_vars=15):
    """Find voltage assignment v[0..14] in {0,...,m-1} such that for every
    constraint c (a tuple of 15 ints), sum(c[i]*v[i] for i in range(15)) % m != 0.
    Returns the list v, or None."""
    # Precompute: for each constraint, which variables it involves (non-zero coeff)
    con_var_idx = []
    con_coeffs = []
    for c in constraints:
        idx = [i for i in range(n_vars) if c[i] != 0]
        con_var_idx.append(idx)
        con_coeffs.append([c[i] for i in idx])

    # Order constraints by number of variables (fewer first = tighter pruning)
    order = sorted(range(len(constraints)), key=lambda i: len(con_var_idx[i]))

    # Backtracking with constraint propagation
    assign = [-1] * n_vars  # -1 means unassigned

    # Precompute for each variable, which constraints involve it
    var_to_cons = [[] for _ in range(n_vars)]
    for ci, idx in enumerate(con_var_idx):
        for vi in idx:
            var_to_cons[vi].append(ci)

    def check_partial(ci, assign):
        """Check constraint ci given partial assignment. Returns:
        True if satisfied (and all vars assigned),
        False if violated (all vars assigned but sum == 0),
        None if undetermined (some vars unassigned)."""
        idx = con_var_idx[ci]
        coeffs = con_coeffs[ci]
        s = 0
        for vi, cf in zip(idx, coeffs):
            av = assign[vi]
            if av == -1:
                return None
            s = (s + cf * av) % m
        return (s != 0)

    # Simple backtracking: assign variables in order 0..14
    def bt(var_idx):
        if var_idx == n_vars:
            # All assigned; verify all constraints
            for ci in order:
                if not check_partial(ci, assign):
                    return False
            return True

        for val in range(m):
            assign[var_idx] = val
            # Check all constraints involving var_idx
            ok = True
            for ci in var_to_cons[var_idx]:
                r = check_partial(ci, assign)
                if r is False:
                    ok = False
                    break
            if ok and bt(var_idx + 1):
                return True
            assign[var_idx] = -1
        return False

    return bt(0), assign if bt(0) else None

def _solve_voltages_v2(m, constraints, n_vars=15):
    """Faster: iterate over voltage assignments using a smarter approach.
    We try to assign voltages one by one, checking constraints whose variables
    are all assigned."""
    con_var_idx = []
    con_coeffs = []
    for c in constraints:
        idx = [i for i in range(n_vars) if c[i] != 0]
        con_var_idx.append(idx)
        con_coeffs.append([c[i] for i in idx])

    order = sorted(range(len(constraints)), key=lambda i: len(con_var_idx[i]))
    var_to_cons = [[] for _ in range(n_vars)]
    for ci, idx in enumerate(con_var_idx):
        for vi in idx:
            var_to_cons[vi].append(ci)

    assign = [-1] * n_vars

    def bt(var_idx):
        if var_idx == n_vars:
            return True
        for val in range(m):
            assign[var_idx] = val
            ok = True
            for ci in var_to_cons[var_idx]:
                idx = con_var_idx[ci]
                coeffs = con_coeffs[ci]
                s = 0
                und = False
                for vi, cf in zip(idx, coeffs):
                    av = assign[vi]
                    if av == -1:
                        und = True
                        break
                    s = (s + cf * av) % m
                if not und and s == 0:
                    ok = False
                    break
            if ok and bt(var_idx + 1):
                return True
            assign[var_idx] = -1
        return False

    if bt(0):
        return assign
    return None

# ---------------------------------------------------------------------------
# Build cover graph from voltages
# ---------------------------------------------------------------------------
def _build_cover(m, voltages):
    """Build the 10*m vertex cover graph. Vertex (u, h) has edges to
    (v, (h + voltages[eid]) % m) for each edge (u, v, eid) of the Petersen."""
    n = N_BASE * m
    edges = []
    for u in range(N_BASE):
        for nb in PETERSEN_ADJ[u]:
            if u < nb:
                eid = _EDGE_ID[(u, nb)]
                vol = voltages[eid]
                for h in range(m):
                    u_h = u * m + h
                    v_h = nb * m + (h + vol) % m
                    edges.append((u_h, v_h))
    return n, edges

# ---------------------------------------------------------------------------
# Girth verification (non-backtracking walks from every vertex)
# ---------------------------------------------------------------------------
def _girth_ok_bfs(adj, g):
    """Return True if the graph has no cycle shorter than g."""
    n = len(adj)
    if n == 0:
        return True
    if g <= 2:
        return True

    max_bfs_depth = (g - 1) // 2

    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = [s]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= max_bfs_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)

        for u in range(n):
            if dist[u] == -1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    continue
                if parent[u] == v or parent[v] == u:
                    continue
                cycle_len = dist[u] + dist[v] + 1
                if cycle_len < g:
                    return False
    return True

def _girth_ok_nb(adj, g):
    """Non-backtracking walk girth check. True iff no cycle shorter than g."""
    n = len(adj)
    if n == 0 or g <= 2:
        return True
    max_depth = g - 1
    for s in range(n):
        frontier = set()
        for v in adj[s]:
            frontier.add((v, s))
        for depth in range(1, max_depth):
            for (v, p) in frontier:
                if v == s:
                    return False
            nxt = set()
            for (v, p) in frontier:
                for w in adj[v]:
                    if w != p:
                        nxt.add((w, v))
            frontier = nxt
            if not frontier:
                break
    return True

# ---------------------------------------------------------------------------
# Fallback: Lazebnik-Ustimenko-Woldar
# ---------------------------------------------------------------------------
def _luw_fallback(k, g):
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

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def construct(k, g):
    deadline = time.time() + 115.0

    # Moore bound: smallest n such that n >= 2 + k * sum_{i=0}^{floor((g-2)/2)-1} (k-1)^i
    # For girth g, the Moore bound for (k, g)-cage is:
    #   g even:  n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
    #   g odd:   n >= k * sum_{i=0}^{(g-3)/2 - 1} (k-1)^i + k * (k-1)^{(g-3)/2}
    # Simplified: n >= 2 + k * sum_{i=0}^{floor((g-2)/2)-1} (k-1)^i
    if g % 2 == 0:
        h = g // 2 - 1
        moore = 2 * sum((k - 1) ** i for i in range(h + 1))
    else:
        h = (g - 3) // 2
        moore = k * sum((k - 1) ** i for i in range(h + 1))
    # n must be even (since n*k must be even)
    if moore % 2 != 0:
        moore += 1

    # For Petersen cover: n = 10 * m
    # Find smallest m with 10*m >= moore
    m_min = max(1, (moore + 9) // 10)

    # Try Petersen cyclic covers
    constraints = _get_constraints(g)
    n_cons = len(constraints)

    # For speed, pre-filter: for each constraint, the net coefficient sum
    # must be non-zero (otherwise the constraint is trivially violated)
    valid_constraints = []
    for c in constraints:
        if sum(c) != 0 or True:  # even if sum is 0, it's still a constraint
            valid_constraints.append(c)
    constraints = valid_constraints
    n_cons = len(constraints)

    best_n = None
    best_edges = None

    # Try m values from m_min upward
    for m in range(m_min, m_min + 200):
        if time.time() > deadline:
            break
        n = N_BASE * m
        if n < moore:
            continue

        # Try to find voltages
        # For small m, brute-force with backtracking is feasible
        # For larger m, use random + local search

        found = False
        voltages = None

        # Strategy 1: backtracking for small m (m <= 20 or so)
        if m <= 30 and n_cons < 2000:
            voltages = _solve_voltages_v2(m, constraints)
            if voltages is not None:
                found = True

        # Strategy 2: random assignments with constraint checking
        if not found:
            rng = random.Random(m * 1000 + 42)
            max_tries = max(500, 5000 // max(1, n_cons // 100))
            for attempt in range(max_tries):
                if time.time() > deadline:
                    break
                vols = [rng.randint(0, m - 1) for _ in range(E_BASE)]
                # Check all constraints
                ok = True
                for c in constraints:
                    s = 0
                    for i in range(E_BASE):
                        if c[i] != 0:
                            s = (s + c[i] * vols[i]) % m
                    if s == 0:
                        ok = False
                        break
                if ok:
                    voltages = vols
                    found = True
                    break

        # Strategy 3: local search / hill climbing
        if not found:
            rng = random.Random(m * 7 + 13)
            vols = [rng.randint(0, m - 1) for _ in range(E_BASE)]
            # Count violations
            def count_violations(v):
                cnt = 0
                for c in constraints:
                    s = 0
                    for i in range(E_BASE):
                        if c[i] != 0:
                            s = (s + c[i] * v[i]) % m
                    if s == 0:
                        cnt += 1
                return cnt

            cur_viol = count_violations(vols)
            if cur_viol == 0:
                voltages = vols
                found = True
            else:
                for it in range(5000):
                    if time.time() > deadline:
                        break
                    if cur_viol == 0:
                        voltages = vols
                        found = True
                        break
                    # Pick a violated constraint, change one of its variables
                    violated = []
                    for c in constraints:
                        s = 0
                        for i in range(E_BASE):
                            if c[i] != 0:
                                s = (s + c[i] * vols[i]) % m
                        if s == 0:
                            violated.append(c)
                    if not violated:
                        voltages = vols
                        found = True
                        break
                    c = violated[rng.randint(0, len(violated) - 1)]
                    var_idx = [i for i in range(E_BASE) if c[i] != 0]
                    vi = var_idx[rng.randint(0, len(var_idx) - 1)]
                    old = vols[vi]
                    vols[vi] = rng.randint(0, m - 1)
                    if vols[vi] == old:
                        vols[vi] = (old + 1) % m
                    new_viol = count_violations(vols)
                    if new_viol <= cur_viol:
                        cur_viol = new_viol
                    else:
                        vols[vi] = old  # revert

        if found and voltages is not None:
            n_cover, edges = _build_cover(m, voltages)
            # Verify
            adj = [[] for _ in range(n_cover)]
            seen = set()
            for u, v in edges:
                if (u, v) in seen or (v, u) in seen or u == v:
                    continue
                adj[u].append(v)
                adj[v].append(u)
                seen.add((u, v))
                seen.add((v, u))
            # Check degree
            deg_ok = True
            for u in range(n_cover):
                if len(adj[u]) != k:
                    deg_ok = False
                    break
            if deg_ok:
                if _girth_ok_bfs(adj, g):
                    if best_n is None or n_cover < best_n:
                        best_n = n_cover
                        best_edges = edges
                    # Return immediately if we found a good one
                    return n_cover, edges

    if best_n is not None:
        return best_n, best_edges

    # Fallback: LUW construction
    return _luw_fallback(k, g)
