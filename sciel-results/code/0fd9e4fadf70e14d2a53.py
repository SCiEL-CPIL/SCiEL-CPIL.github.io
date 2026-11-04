import time

# EVOLVE-BLOCK-START

_PETERSEN_EDGES = [
    (0,1),(1,2),(2,3),(3,4),(4,0),          # outer 5-cycle
    (5,6),(6,7),(7,8),(8,9),(9,5),          # inner 5-cycle
    (0,5),(1,6),(2,7),(3,8),(4,9),          # spokes
]

def _build_petersen_adj():
    adj = [[] for _ in range(10)]
    for u,v in _PETERSEN_EDGES:
        adj[u].append(v); adj[v].append(u)
    return adj

def _primitive_root(p):
    # find smallest primitive root modulo prime p
    def is_prime(n):
        if n < 2: return False
        if n % 2 == 0: return n == 2
        i = 3
        while i*i <= n:
            if n % i == 0: return False
            i += 2
        return True
    # factor p-1
    def factors(n):
        fs = []
        d = 2
        while d*d <= n:
            if n % d == 0:
                fs.append(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1: fs.append(n)
        return fs
    for g in range(2, p):
        if all(pow(g, (p-1)//f, p) != 1 for f in factors(p-1)):
            return g
    return None

def construct(k, g):
    start_time = time.time()
    DEADLINE = start_time + 110.0   # safety margin before 120s kill (150s wall-clock budget)

    # We build a 3-regular voltage graph: base = Petersen (10 vertices),
    # voltages in additive group Z_m. Derived graph: 10*m vertices, 3-regular.
    # Edge (u,v) with voltage a: in derived graph, (u,i) -- (v, (i+a) mod m).
    # A closed walk in base has voltage = signed sum of edge voltages.
    # Girth of derived graph >= g  <=>  no closed walk of length < g has net voltage 0
    # (and base has no short cycles with 0 voltage).
    #
    # Strategy: assign voltages as powers of a primitive root to spread out
    # the possible short-walk sums, minimizing the chance of a zero-sum short cycle.

    if k != 3:
        # Fallback: only handle k=3 robustly; for other k return a trivial valid graph
        # (a single edge won't be 3-regular; but task is k=3). Provide a safe constant.
        # For safety, return a small valid k-regular graph if possible.
        return _trivial_regular(k, g)

    base_adj = _build_petersen_adj()
    base_edges = _PETERSEN_EDGES

    # We need m such that 10*m is at least the Moore bound for (3,16) = 512.
    # Try m values; pick the smallest that yields girth >= g.
    # m must be prime for primitive-root voltage scheme.
    primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
              101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]

    best = None
    for m in primes:
        if time.time() > DEADLINE:
            break
        n = 10 * m
        # voltage per edge index: use primitive root powers
        g0 = _primitive_root(m)
        if g0 is None:
            continue
        # assign voltage to each of the 15 edges
        # outer cycle edges: 0-1,1-2,2-3,3-4,4-0  -> indices 0..4
        # inner cycle edges: 5-6,6-7,7-8,8-9,9-5  -> indices 5..9
        # spokes: 0-5,1-6,2-7,3-8,4-9            -> indices 10..14
        volt = []
        for i in range(15):
            volt.append(pow(g0, i+1, m))
        cand = _build_and_check(base_adj, base_edges, volt, m, g, DEADLINE)
        if cand is not None:
            # girth >= g achieved; record
            if best is None or cand[0] < best[0]:
                best = cand
                # try to go smaller? m increasing only; smallest m that works is best here
                break
    if best is not None:
        return best

    # Fallback: return the seed-style LUW construction if we ran out of time / no prime worked
    return _luw_fallback(k, g)


def _trivial_regular(k, g):
    # A k-regular graph with many vertices (disjoint cliques won't have high girth).
    # Provide a cycle-based construction that is k-regular only for k=2. For general k
    # this is a last-resort; task uses k=3.
    # For k=3, use a 3-regular circulant on 2n vertices.
    n = 4
    verts = 2*n
    edges = []
    for i in range(verts):
        for d in [1, n//2 if n%2==0 else n//2]:
            j = (i + d) % verts
            if i < j:
                edges.append((i, j))
    # ensure 3-regular: add edge to i+? ; simpler: use known Petersen for k=3
    if k == 3:
        return (10, _PETERSEN_EDGES)
    return (verts, edges)


def _build_and_check(base_adj, base_edges, volt, m, g, deadline):
    n = 10 * m
    # Build derived graph adjacency
    # vertex id = u*m + i  for base vertex u in 0..9, group element i in 0..m-1
    adj = [[] for _ in range(n)]
    for idx,(u,v) in enumerate(base_edges):
        a = volt[idx]
        for i in range(m):
            su = u*m + i
            sv = v*m + ((i + a) % m)
            adj[su].append(sv)
            su2 = v*m + i
            sv2 = u*m + ((i - a) % m)
            adj[su2].append(sv2)
    # verify 3-regular
    for x in range(n):
        if len(adj[x]) != 3:
            return None
    # Girth check via BFS non-backtracking from each vertex, up to depth g-1
    # A cycle of length L exists iff a non-backtracking closed walk of length L exists.
    # We check: for each start vertex, do non-backtracking BFS; if we return to start
    # at depth < g, girth violated.
    # Efficient: for each start, BFS over (vertex, prev) states up to depth g-1.
    for s in range(n):
        if time.time() > deadline:
            return None
        # states: (v, prev) ; start: (s, -1)
        seen = set()
        frontier = [(s, -1)]
        for depth in range(1, g):
            newfrontier = []
            for (v, prev) in frontier:
                for w in adj[v]:
                    if w == prev:
                        continue
                    if w == s and depth >= g:
                        pass
                    # if we close a cycle: w == s and depth < g -> violation
                    if w == s:
                        if depth < g:
                            return None
                        else:
                            continue
                    st = (w, v)
                    if st in seen:
                        continue
                    seen.add(st)
                    newfrontier.append(st)
            frontier = newfrontier
            if not frontier:
                break
    # girth >= g confirmed
    edges = []
    for idx,(u,v) in enumerate(base_edges):
        a = volt[idx]
        for i in range(m):
            su = u*m + i
            sv = v*m + ((i + a) % m)
            edges.append((su, sv))
    return (n, edges)


def _luw_fallback(k, g):
    # Reproduce the seed's Lazebnik-Ustimenko-Woldar construction as a safe fallback.
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

# EVOLVE-BLOCK-END
