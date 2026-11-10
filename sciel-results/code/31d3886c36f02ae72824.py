"""Search for a small 3-regular graph of girth >= g via algebraic circulant constructions,
falling back to LUW construction.

Strategy: systematically enumerate doubly-circulant (3,16) cage candidates over Z_m and
Z_m x Z_2, certify girth exactly via non-backtracking walk enumeration (O(n*g)), and
return the smallest valid graph found.  This replaces random configuration-model sampling
(which cannot succeed at n~512-936 within 120 s) with a deterministic algebraic search
over a rich family of known small-cage constructions.
"""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# LUW construction (fallback)
# ---------------------------------------------------------------------------
def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k; returns (n, edges)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
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
# Fast girth certification via non-backtracking walk enumeration
# ---------------------------------------------------------------------------
def _girth_ok_nb(adj, n, g):
    """Return True iff the graph has girth >= g, certified exactly.

    Uses non-backtracking walk enumeration from every vertex up to length g-1.
    A cycle of length L < g corresponds to a non-backtracking closed walk of
    length L (each cycle has exactly 2L non-backtracking closed walks of length L,
    one for each direction, starting at each of its L vertices -- but we just need
    to detect ANY closed non-backtracking walk of length < g).

    For efficiency: from each vertex, enumerate all non-backtracking walks of
    length up to g-1. If a walk returns to its start vertex, a cycle of length
    <= g-1 exists.

    State: (current_vertex, previous_vertex) -> set of start vertices that can
    reach this state via a non-backtracking walk of the current length.
    """
    if g <= 2:
        return True

    # For each start vertex s, we track the set of (v, u) states reachable
    # by non-backtracking walks from s.  A cycle exists iff (s, prev) is
    # reachable for some prev (i.e., we returned to s).
    #
    # More efficient: for each start s, do a BFS over states (v, u) where
    # u is the vertex we came from (so we don't backtrack).  Start states:
    # (s, -1) -- but -1 is a sentinel.  Actually, standard NB walk:
    #   length 1: (s, w) for each neighbor w of s.
    #   length L: (v, u) -> (w, v) for each neighbor w of v, w != u.
    #   Cycle detected if v == s at any point (and length >= 3 to avoid trivial).
    #
    # Even better: we can do this simultaneously for all starts using
    # bitsets.  But for n up to ~1000 and g=16, per-start BFS is fine:
    #   states per start: at most 3 * 2^(g-2) ~ 3*8192 = 24576, but pruned heavily.
    #   Actually the number of NB walks of length L in a 3-regular graph is
    #   at most 3 * 2^(L-1).  For L=15: 3*8192 = 24576.  For n=1000 starts:
    #   ~25M state visits, each O(1) with adjacency lookup.  Should be < 1s.

    # Precompute adjacency as frozensets for fast lookup
    adj_sets = [frozenset(a) for a in adj]

    for s in range(n):
        # BFS over NB-walk states (v, u) where u is previous vertex
        # visited states to avoid reprocessing
        # Use a set of (v, u) tuples
        visited = set()
        # Initial states: (s, -1) is the "start" but we process length 1 first
        # Actually, let's use a queue of (v, u, length)
        # Start: for each neighbor w of s, state (w, s) at length 1
        queue = []
        for w in adj_sets[s]:
            queue.append((w, s, 1))
            visited.add((w, s))

        qi = 0
        while qi < len(queue):
            v, u, length = queue[qi]
            qi += 1
            if length >= g:
                continue  # no need to extend further
            # Extend: for each neighbor w of v, w != u
            for w in adj_sets[v]:
                if w == u:
                    continue
                # Check if we've closed a cycle: w == s and length+1 >= 3
                if w == s and length + 1 >= 3:
                    return False
                state = (w, v)
                if state not in visited:
                    visited.add(state)
                    if length + 1 < g:  # only enqueue if we'll extend
                        queue.append((w, v, length + 1))
        # If we get here, no cycle of length < g found from s
    return True


# ---------------------------------------------------------------------------
# Circulant graph constructions
# ---------------------------------------------------------------------------
def _circulant_graph(n, offsets):
    """Build a circulant graph on Z_n with the given symmetric offset set.

    offsets must be a set of integers mod n such that for every d in offsets,
    -d mod n is also in offsets (symmetry), and 0 not in offsets.
    Returns (n, edges) or None if not simple/valid.
    """
    offsets = set(offsets)
    # Remove 0
    offsets.discard(0)
    # Ensure symmetry
    sym_offsets = set()
    for d in offsets:
        sym_offsets.add(d % n)
        sym_offsets.add((-d) % n)
    sym_offsets.discard(0)
    if len(sym_offsets) != len(offsets) * 2 // (1 if any(d == -d % n for d in offsets) else 1):
        # Just rebuild properly
        pass
    # Rebuild: the degree is |sym_offsets|
    if len(sym_offsets) != 3:
        return None  # we need degree 3
    edges = []
    for i in range(n):
        for d in sym_offsets:
            j = (i + d) % n
            if i < j:
                edges.append((i, j))
    # Verify degree
    deg = [0] * n
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    if any(d != 3 for d in deg):
        return None
    # Verify simple
    if len(edges) != n * 3 // 2:
        return None
    return n, edges


def _doubly_circulant_graph(m, a, b, c):
    """Build a doubly-circulant 3-regular graph on Z_m x {0,1}.

    Vertices: (i, s) for i in Z_m, s in {0,1}.
    Edges:
      - Within same side: (i, s) -- (i+a, s)  [cyclic, degree 2 per side if a != 0 and 2a != 0 mod m]
        Actually, a single offset a gives degree 2 (forward and backward) if a != -a mod m.
        If a == -a mod m (i.e., 2a = 0 mod m), then it's a single edge type but still
        (i,s)-(i+a,s) and (i,s)-(i-a,s) = (i,s)-(i+a,s) so degree 1.  We need degree 2
        from the within-side edges, so we need a such that a != -a mod m, OR use two
        offsets.  Let's use a single offset a with a not self-inverse, giving degree 2.
      - Between sides: (i, 0) -- (i+b, 1) and (i, 0) -- (i+c, 1)  [two cross edges per vertex]

    Total degree: 2 (within) + 2 (cross) = 4.  That's too many.

    Let me reconsider: for 3-regular doubly circulant:
      - Within side: one offset a, giving degree 2 (if a != -a) or 1 (if a = -a).
      - Cross: one offset b, giving degree 1 per vertex.
      Total: 3 if a != -a.

    So: edges are (i,s)-(i+a,s) and (i,0)-(i+b,1).
    Degree: 2 + 1 = 3. Good.

    Constraints for simplicity:
      - a != 0 mod m
      - 2a != 0 mod m (so that a and -a are distinct, giving 2 distinct neighbors)
      - b can be anything mod m
      - No multi-edges: need (i+a, s) != (i, s) i.e. a != 0, and cross edges
        (i+b, 1) != (i, 1) -- but cross edges connect different sides so no issue.
      - Also need that the cross edge doesn't coincide with a within-side edge,
        which it can't since they connect different sides.
      - Need m >= 3 (for a to have a != -a mod m, need m >= 3 and a not = m/2 if m even)
    """
    n = 2 * m
    edges = []
    # Within-side edges: (i, s) -- (i+a, s) for s in {0,1}
    for s in (0, 1):
        for i in range(m):
            j = (i + a) % m
            # Add edge once: only if i < j in the global vertex index
            vi = s * m + i
            vj = s * m + j
            if vi < vj:
                edges.append((vi, vj))
    # Cross edges: (i, 0) -- (i+b, 1)
    for i in range(m):
        vi = i  # side 0
        vj = m + ((i + b) % m)  # side 1
        if vi < vj:
            edges.append((vi, vj))
        else:
            edges.append((vj, vi))

    # Verify
    if len(edges) != n * 3 // 2:
        return None
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != 3 for d in deg):
        return None
    # Check for multi-edges
    if len(set(tuple(sorted(e)) for e in edges)) != len(edges):
        return None
    return n, edges


def _build_adj(edges, n):
    """Build adjacency list from edge list."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _search_circulants(k, g, deadline):
    """Search circulant and doubly-circulant 3-regular graphs for girth >= g.

    Returns (n, edges) for the smallest valid graph found, or None.
    """
    best = None
    best_n = float('inf')

    # --- Doubly-circulant graphs on Z_m x {0,1} ---
    # n = 2m.  We want n as small as possible but >= Moore bound.
    # For k=3, g=16: Moore bound ~ 2*(2^7 - 1) + 2 = 254? Let's compute properly.
    # Moore bound for (k,g) with g even: n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2 * (k^(g/2) - 1)/(k-1)
    # For k=3, g=16: n >= 2 * (3^8 - 1)/2 = 3^8 - 1 = 6560.  Wait, that's for the Moore bound
    # of a (3,16)-cage.  Actually the Moore bound is:
    #   n >= 2 * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(g/2-1))
    # For k=3, g=16: g/2-1 = 7, so sum = 1 + 2 + 4 + 8 + 16 + 32 + 64 + 128 = 255
    # n >= 2 * 255 = 510.
    # So n >= 510.  With n = 2m, m >= 255.

    moore = 2 * sum((k - 1)**i for i in range(g // 2))
    # For k=3, g=16: moore = 2 * (1+2+4+...+128) = 2*255 = 510

    # Search m from ceil(moore/2) upward
    m_lo = max(3, (moore + 1) // 2)
    m_hi = m_lo + 300  # search a window

    for m in range(m_lo, m_hi + 1):
        if time.time() > deadline:
            break
        # Try various (a, b) pairs
        # a must satisfy: a != 0, 2a != 0 mod m
        # b can be anything
        # To keep the search tractable, try a from 1 to m-1 (skip 0 and m/2 if m even)
        # and b from 0 to m-1
        # That's O(m^2) per m, which for m=255 is ~65k pairs.  Each girth check is
        # O(n * 3 * 2^(g-2)) which is too slow for 65k graphs.
        #
        # Optimization: use a fast necessary condition first.
        # The girth of a doubly-circulant graph can be bounded by the orders
        # of a and b in Z_m.  A cycle of length L corresponds to a relation
        # among the generators.  Specifically, the graph is a Cayley graph of
        # Z_m x Z_2 with generators {(a,0), (0,1), (b,1)} -- wait, let me re-derive.
        #
        # Actually, the vertices are (i, s).  The edges are:
        #   (i,s) -- (i+a, s): this is translation by (a, 0)
        #   (i,0) -- (i+b, 1): this is translation by (b, 1) from side 0,
        #     and by (-b, 1) from side 1 (since (i,1) -- (i-b, 0) = (i + (-b), 1+1 mod 2) = (i-b, 0))
        #
        # So the generators are: (a, 0), (b, 1), (-b, 1).  Note (b,1) and (-b,1) are
        # distinct unless 2b = 0 mod m.
        #
        # The group is Z_m x Z_2.  The Cayley graph has generators S = {(a,0), (b,1), (-b,1)}.
        # For this to be 3-regular, we need |S| = 3 and no generator is the identity,
        # and no generator is the inverse of another (or if it is, we'd have degree 2).
        # Inverse of (a,0) is (-a, 0).  Inverse of (b,1) is (-b, 1) since (b,1)+(-b,1) = (0,0).
        # So (b,1) and (-b,1) are inverses of each other!  That means the generator set
        # is not closed under inverses in the usual Cayley graph sense -- we have both
        # an element and its inverse, which is fine for an undirected graph, but it means
        # the degree contribution from {(b,1), (-b,1)} is 2 (one for each), and from
        # {(a,0)} is 2 (forward and backward) unless a = -a mod m.
        #
        # Wait, I think I made an error.  Let me recount:
        #   - Within side: (i,s) connects to (i+a, s) AND (i-a, s).  That's 2 neighbors.
        #     This corresponds to generators (a,0) and (-a,0).
        #   - Cross: (i,0) connects to (i+b, 1).  And (i,1) connects to (i-b, 0).
        #     So (i,0) has 1 cross neighbor, (i,1) has 1 cross neighbor.
        #   Total degree: 2 + 1 = 3.  Correct.
        #
        # In Cayley graph terms with group G = Z_m x Z_2:
        #   Generators: {(a,0), (-a,0), (b,1)} -- but (b,1) is its own... no.
        #   (b,1) + (b,1) = (2b, 0).  (b,1) + (-b,1) = (0,0).  So (-b,1) is the inverse of (b,1).
        #   The edge (i,0)-(i+b,1) means (i,0) + (b,1) = (i+b, 1).  And (i,1) + (b,1) = (i+b, 0).
        #   But the cross edge from (i,1) goes to (i-b, 0) = (i,1) + (-b,1).
        #   So the generator set is {(a,0), (-a,0), (b,1), (-b,1)}?  No, that would give degree 4.
        #
        # I think the issue is that the cross edges are not a simple Cayley graph with a
        # fixed generator set because the "direction" of the cross edge depends on the side.
        #
        # Let me just use the direct adjacency and check girth via NB walks.
        # To make the search tractable, I'll use a smarter approach:
        #   1. For each m, try a = 1 (most common in small cages) and vary b.
        #   2. Also try a few other small values of a.
        #   3. Use a fast pre-check: the girth is at least the minimum order of
        #      non-trivial relations among the "step" vectors.

        # Fast pre-check: in the group Z_m x Z_2, a cycle of length L corresponds to
        # a sequence of L steps from the generator set that sums to (0,0).
        # The generators (in terms of translations) are:
        #   Step A: (a, 0)  [within side, forward]
        #   Step A': (-a, 0) [within side, backward]
        #   Step B: (b, 1)  [cross, from side 0 to 1]
        #   Step B': (-b, 1) [cross, from side 1 to 0]
        #
        # But the available steps depend on the current side.  From side 0, we can take
        # A, A', or B.  From side 1, we can take A, A', or B'.
        #
        # A cycle must return to the starting vertex AND side.  So the sum of all
        # (dx, ds) steps must be (0, 0) in Z_m x Z_2.  The ds component: each B or B'
        # contributes 1 mod 2, so the number of cross steps must be even.
        #
        # For a fast necessary condition: the minimum number of steps to get (0,0)
        # in Z_m x Z_2 using the available generators.  This is a shortest path problem
        # in the group, but we can use a simple bound:
        #   - The order of (a,0) in Z_m is m/gcd(a,m).  A cycle using only A and A'
        #     has length = order of (a,0) = m/gcd(a,m).  So girth <= m/gcd(a,m).
        #   - For the girth to be >= g, we need m/gcd(a,m) >= g.
        #
        # This is a great pre-check!  For g=16, we need m/gcd(a,m) >= 16.

        # Try a values that satisfy m/gcd(a,m) >= g
        valid_as = []
        for a in range(1, m):
            if 2 * a % m == 0:
                continue  # a = -a mod m, would give degree 1 within side
            order_a = m // np.gcd(a, m)
            if order_a >= g:
                valid_as.append(a)

        if not valid_as:
            continue

        # For each valid a, try b values
        # To limit the search, try b = 0, 1, 2, ..., min(m-1, 50)
        # and also b = m-1, m-2, etc. (by symmetry, b and m-b give isomorphic graphs)
        b_values = list(range(0, min(m, 30)))
        # Add a few more spread out
        if m > 60:
            b_values.extend([m // 3, m // 2, 2 * m // 3])
        b_values = list(set(b_values))

        for a in valid_as[:10]:  # limit to first 10 valid a's
            if time.time() > deadline:
                break
            for b in b_values:
                if time.time() > deadline:
                    break
                result = _doubly_circulant_graph(m, a, b, 0)
                if result is None:
                    continue
                n, edges = result
                if n >= best_n:
                    continue
                adj = _build_adj(edges, n)
                if _girth_ok_nb(adj, n, g):
                    best = (n, edges)
                    best_n = n
                    # Keep searching for smaller
                    break

    # --- Pure circulant graphs on Z_n ---
    # 3-regular circulant: need offset set of size 3 (symmetric).
    # Options: {a, b, -a} with b = -b (i.e., 2b = 0 mod n) -- then offsets are {a, -a, b}
    #   where b = n/2 (n even).  Degree: 2 (from a, -a) + 1 (from b) = 3.
    #   Or {a, b, c} all distinct, none self-inverse, and {-a, -b, -c} = {a, b, c}.
    #   This requires that the set is closed under negation, so it must contain
    #   pairs {d, -d} and possibly a self-inverse element.  For size 3, we need
    #   one self-inverse element (n even, d = n/2) and one pair {a, -a}.
    #   So offsets = {a, -a, n/2} with a != 0, a != n/2, 2a != 0 mod n.
    #
    # n must be even.  n >= moore.
    n_lo = moore
    if n_lo % 2:
        n_lo += 1
    n_hi = n_lo + 200

    for n in range(n_lo, n_hi + 1, 2):
        if time.time() > deadline:
            break
        if n >= best_n:
            break
        # a ranges over 1..n/2-1, skipping n/2
        # Pre-check: order of a in Z_n is n/gcd(a,n).  Need >= g.
        # Also order of n/2 is 2, which is < g for g >= 4, so cycles using the
        # n/2 edge could be short.  Specifically, (i) -- (i+n/2) -- (i) is a 2-cycle?
        # No, it's a single edge.  But (i) -- (i+n/2) -- (i+n/2+a) -- ... could form
        # short cycles.  The pre-check is less clean here.
        for a in range(1, n // 2):
            if a == n // 2:
                continue
            order_a = n // np.gcd(a, n)
            if order_a < g:
                continue
            # Try offset set {a, -a, n/2}
            offsets = {a, n - a, n // 2}
            result = _circulant_graph(n, offsets)
            if result is None:
                continue
            n_c, edges = result
            if n_c >= best_n:
                continue
            adj = _build_adj(edges, n_c)
            if _girth_ok_nb(adj, n_c, g):
                best = (n_c, edges)
                best_n = n_c
                break

    return best


def _deadline_reached(t0, limit):
    return (time.time() - t0) > limit


def construct(k, g):
    t0 = time.time()
    HARD_LIMIT = 140.0  # safety margin under 150s wall-clock

    # Try algebraic circulant constructions first
    result = _search_circulants(k, g, HARD_LIMIT)
    if result is not None:
        return result

    # Fallback: guaranteed-valid LUW construction
    try:
        return _luw_construct(k, g)
    except Exception:
        raise

# EVOLVE-BLOCK-END
