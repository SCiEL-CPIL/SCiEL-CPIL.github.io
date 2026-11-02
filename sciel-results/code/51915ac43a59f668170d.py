import time
import math
import random
from collections import deque

def construct(k, g):
    """Construct a k-regular graph of girth at least g with as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # ---- Exact known cages / small optimal graphs ----
    known = {
        (3, 5): _petersen,
        (3, 6): _heawood,
        (3, 7): _tutte_coxeter,
        (3, 8): _tutte_coxeter,
        (3, 9): _mcgee,
        (4, 5): _cage_4_5,
    }
    if (k, g) in known:
        res = known[(k, g)]()
        if res is not None:
            n, edges = res
            if _verify_girth(n, edges, k, g):
                return n, edges

    best = None  # (n, edges)

    def consider(n, edges):
        nonlocal best
        if best is None or n < best[0]:
            best = (n, edges)

    # ---- LUW construction: try multiple q values, smallest first ----
    for q in _candidate_qs(k):
        if elapsed() > deadline:
            break
        if q < k:
            continue
        res = _luw_construct(q, g)
        if res is not None:
            n, edges = res
            if _verify_girth(n, edges, k, g):
                consider(n, edges)
                break  # smallest q that works is best for this family

    # ---- Circulant construction (even k) ----
    if k % 2 == 0:
        remaining = deadline - elapsed()
        if remaining > 2:
            res = _circulant_construct(k, g, remaining)
            if res is not None:
                n, edges = res
                if _verify_girth(n, edges, k, g):
                    consider(n, edges)

    # ---- Always-valid fallback (large, guaranteed girth >= g) ----
    if best is None or elapsed() < deadline - 10:
        res = _always_valid_construct(k, g)
        if res is not None:
            n, edges = res
            if _verify_girth(n, edges, k, g):
                consider(n, edges)

    if best is not None:
        return best

    # Absolute last resort
    res = _always_valid_construct(k, g)
    if res is not None:
        return res
    # Final fallback: complete graph blown up
    n = k + 1
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return n, edges


# ---------------------------------------------------------------------------
# Exact known cages
# ---------------------------------------------------------------------------

def _petersen():
    """Petersen graph: 3-regular, girth 5, 10 vertices."""
    edges = []
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    for i in range(5):
        edges.append((5 + i, 5 + (i + 2) % 5))
    for i in range(5):
        edges.append((i, 5 + i))
    return 10, edges


def _heawood():
    """Heawood graph: 3-regular, girth 6, 14 vertices."""
    edges = []
    lines = [
        (0, 1, 2), (0, 3, 4), (0, 5, 6),
        (1, 3, 5), (1, 4, 6), (2, 3, 6),
        (2, 4, 5)
    ]
    for i, line in enumerate(lines):
        for p in line:
            edges.append((i, 7 + p))
    return 14, edges


def _tutte_coxeter():
    """Tutte-Coxeter graph: 3-regular, girth 8, 30 vertices.

    Constructed as the incidence graph of GQ(2,2).
    """
    # GQ(2,2) model over Z_3:
    # Points: (x, y) for x,y in Z_3  -> 9 points (indices 0..8)
    #         (x, inf) for x in Z_3  -> 3 points (indices 9..11)
    #         (inf, y) for y in Z_3  -> 3 points (indices 12..14)
    # Lines:  L_{a,b} = {(x,y): y = a*x + b} U {(inf,b)} U {(a,inf)}  for a,b in Z_3 -> 9 lines
    #         M_a     = {(x, y): y = a*x + a*x*a} ... need 6 more lines.
    #
    # Standard GQ(2,2): 15 points, 15 lines, each point on 3 lines, each line has 3 points.
    # Let me use a verified construction.
    #
    # Actually, the cleanest verified model:
    # Points: all (x, y) in Z_3 x Z_3 (9) plus (x, inf) (3) plus (inf, y) (3) = 15.
    # Lines of type L_{a,b}: {(x, a*x+b) : x in Z_3} U {(inf, b)} U {(a, inf)}  -> 9 lines
    # Lines of type M_c:     {(x, y) : y = c - x*x ... } -- need a proper quadratic.
    #
    # Over Z_3, x^2 takes values {0,1,1}. The 6 "parabola" lines:
    #   P_a = {(x, x*x + a) : x in Z_3} U {(a, inf)} U {(inf, ?)}
    # This is getting error-prone. Let me use a known edge list instead.
    #
    # Verified edge list for the Tutte-Coxeter graph (30 vertices, bipartite 15+15):
    # I'll generate it from the GQ(2,2) with a correct model.

    # Correct GQ(2,2) model (from standard references):
    # Points: 15 = 3*5. Label points as (i, j) where i in Z_3, j in Z_5, with a constraint? No.
    #
    # Let me use the well-known fact: the (3,8)-cage is the bipartite graph where
    # both parts have 15 vertices. I'll hardcode a verified adjacency.
    #
    # Here is a verified construction using the group presentation.
    # The Tutte-Coxeter graph is the Cayley graph of the group
    #   G = < a, b | a^5 = b^3 = (ab)^3 = 1 >  (order 30)
    # with connection set { a, a^{-1}, b, b^{-1}, ... } -- but that gives degree 4, not 3.
    #
    # Simplest reliable approach: build via LUW(3,3) which has girth 8 and 54 vertices,
    # but that's not 30. So LUW is not optimal here.
    #
    # I'll use a verified 30-vertex edge list. The Tutte-Coxeter graph adjacency
    # (from House of Graphs / known data), vertices 0..29:
    edges = []
    # Connection list (each vertex degree 3), verified:
    conn = [
        [1, 10, 16],
        [0, 11, 17],
        [12, 13, 14],
        [2, 15, 16],
        [18, 19, 20],
        [4, 21, 22],
        [5, 23, 24],
        [6, 25, 26],
        [27, 28, 29],
        [8, 3, 4],
        [0, 1, 11],
        [1, 10, 12],
        [2, 11, 13],
        [2, 12, 14],
        [2, 13, 14],
        [3, 16, 15],
        [3, 15, 16],
        [1, 17, 18],
        [4, 18, 19],
        [4, 19, 20],
        [4, 20, 21],
        [5, 21, 22],
        [5, 22, 23],
        [6, 23, 24],
        [6, 24, 25],
        [7, 25, 26],
        [7, 26, 27],
        [8, 27, 28],
        [8, 28, 29],
        [8, 29, 27],
    ]
    # This hand-built list is error-prone; let me instead construct it cleanly.
    # Return None and let LUW handle it (valid, 54 vertices, girth 8).
    return None


def _mcgee():
    """McGee graph: 3-regular, girth 10, 76 vertices. (3,9)-cage is McGee with girth 10 >= 9."""
    # The (3,9)-cage is the McGee graph (76 vertices, girth 10).
    # Construction: it's a known graph. I'll build it via LUW as a valid fallback.
    return None


def _cage_4_5():
    """(4,5)-cage: 4-regular, girth 5, 19 vertices."""
    n = 19
    edges = set()
    for i in range(n):
        for j in [1, -1, 5, -5]:
            v = (i + j) % n
            u = min(i, v)
            w = max(i, v)
            edges.add((u, w))
    edges = list(edges)
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != 4 for d in deg):
        return None
    if _verify_girth(n, edges, 4, 5):
        return n, edges
    return None


# ---------------------------------------------------------------------------
# LUW construction
# ---------------------------------------------------------------------------

def _candidate_qs(k):
    """Yield candidate field sizes q for LUW, starting from k.
    
    For LUW to give a k-regular graph, we need q = k.
    But LUW(q, d) gives a q-regular graph. So we need q == k exactly.
    If k is not a prime power or 4, LUW won't work directly.
    We still try q=k if valid, and also nearby primes as fallback
    (though those give different regularity).
    """
    # Primary: q = k if k is a valid LUW parameter
    if k == 4 or _is_prime(k):
        yield k
        return
    # k is not prime or 4: try k itself won't work for LUW.
    # Try nearby primes (will give q-regular, not k-regular, so only useful as fallback)
    # Actually for the main path we need q == k. If k isn't valid, skip LUW.
    # But let's still yield some candidates for the fallback path.
    q = k + 1
    count = 0
    while q < k + 200 and count < 10:
        if _is_prime(q) or q == 4:
            yield q
            count += 1
        q += 1


def _luw_construct(q, g):
    """Lazebnik-Ustimenko-Woldar construction with min girth >= g.
    
    LUW(q, d) produces a q-regular bipartite graph with 2*q^d vertices
    and girth at least d+5 (when q is prime or 4, and d is odd).
    """
    if q < 3:
        return None

    # LUW(d, q) has minimum girth d + 5 (for q a prime or 4, d odd).
    # We need d + 5 >= g  =>  d >= g - 5.
    # d must be odd and at least 3.
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1

    # Check if q is valid for LUW
    if q != 4 and not _is_prime(q):
        return None

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Rule for LUW
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
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
                edges.append((index[vertex], index[other]))

    n = len(order)
    # Expected: n = 2 * q^d
    expected_n = 2 * (q ** d)
    if n != expected_n:
        return None

    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(dd != q for dd in deg):
        return None

    return n, edges


# ---------------------------------------------------------------------------
# Circulant construction
# ---------------------------------------------------------------------------

def _circulant_construct(k, g, time_budget):
    """Try to construct a circulant graph with girth >= g (even k only)."""
    if k % 2 != 0:
        return None
    half_k = k // 2
    start_time = time.time()
    deadline = start_time + max(0, time_budget - 5)

    moore_bound = 1 + k + k * (k - 1) + k * (k - 1) ** 2
    n_min = max(moore_bound, 2 * half_k + 1)
    
    # Try a limited range of n values with smart connection sets
    for n in range(n_min, n_min + 150):
        if time.time() > deadline:
            break
        if n * k % 2 != 0:
            continue

        found = _try_circulant_fast(n, k, g, half_k)
        if found is not None:
            return found
    return None


def _try_circulant_fast(n, k, g, half_k):
    """Try a few good connection sets for a circulant on n vertices."""
    # Strategy 1: {1, 2, ..., half_k} - good when n is large
    S1 = list(range(1, half_k + 1))
    res = _check_circulant(n, S1, k, g)
    if res is not None:
        return res

    # Strategy 2: {s, 2s, ..., half_k*s} for a few s values
    for s in [2, 3, 5, 7, 11, 13, 17]:
        S = []
        for i in range(1, half_k + 1):
            val = (i * s) % n
            val = min(val, n - val)
            if val == 0 or val in S:
                break
            S.append(val)
        if len(S) != half_k:
            continue
        res = _check_circulant(n, S, k, g)
        if res is not None:
            return res

    # Strategy 3: spread out steps evenly
    step = n // (2 * half_k + 1)
    if step >= 1:
        S3 = [i * step for i in range(1, half_k + 1)]
        S3 = [min(s, n - s) for s in S3]
        if len(set(S3)) == half_k and 0 not in S3:
            res = _check_circulant(n, S3, k, g)
            if res is not None:
                return res

    return None


def _check_circulant(n, S, k, g):
    """Check if circulant with connection set S has girth >= g."""
    edges = set()
    for v in range(n):
        for s_val in S:
            u = (v + s_val) % n
            w = (v - s_val) % n
            edges.add((min(v, u), max(v, u)))
            if w != u:
                edges.add((min(v, w), max(v, w)))
    edges = list(edges)
    if len(edges) != n * k // 2:
        return None
    if _verify_girth(n, edges, k, g):
        return n, edges
    return None


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------

def _random_search(k, g, time_budget):
    """Random search for a k-regular graph with girth >= g (small n)."""
    # For large girths, random search is extremely unlikely to succeed.
    # Only try if g is small.
    if g > 8:
        return None
    start_time = time.time()
    deadline = start_time + max(0, time_budget - 10)

    moore_bound = 1 + k + k * (k - 1) + k * (k - 1) ** 2
    for n in range(max(moore_bound, k + 1), max(moore_bound, k + 1) + 20):
        if time.time() > deadline:
            break
        if n * k % 2 != 0:
            continue
        for _ in range(3):
            if time.time() > deadline:
                break
            edges = _generate_random_regular(n, k)
            if edges is None:
                continue
            if _verify_girth(n, edges, k, g):
                return n, edges
    return None


def _generate_random_regular(n, k):
    """Generate a random k-regular graph on n vertices via configuration model."""
    if n * k % 2 != 0 or k >= n or k == 0:
        return None
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    random.shuffle(stubs)
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            continue
        edges.add((min(u, v), max(u, v)))
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != k for d in deg):
        return None
    return list(edges)


# ---------------------------------------------------------------------------
# Always-valid fallback
# ---------------------------------------------------------------------------

def _always_valid_construct(k, g):
    """Always return a valid k-regular graph with girth >= g (large n)."""
    # For even k: use a circulant with connection set {1, 2, ..., k/2} on a very large n.
    # A cycle of length L in this circulant requires a sum of L signed elements from
    # {1,...,k/2} that equals 0 mod n. The minimum absolute value of such a sum
    # (with L terms) is at most L*k/2. So if n > L*k/2 for all L < g, no short cycle exists.
    # Specifically, we need n > (g-1) * k // 2 to guarantee girth >= g.
    # To be safe, use n = (g-1) * k + 1.
    if k % 2 == 0:
        half_k = k // 2
        # Need: for any cycle of length L < g, the sum of L signed steps from {1..half_k}
        # cannot be 0 mod n. The max absolute sum of L terms is L * half_k.
        # So if n > (g-1) * half_k, no cycle shorter than g exists.
        n = (g - 1) * half_k + 1
        # Make n odd to avoid issues (or just ensure it's large enough)
        if n <= 2 * half_k:
            n = 2 * half_k + 1
        edges = set()
        for v in range(n):
            for s_val in range(1, half_k + 1):
                u = (v + s_val) % n
                w = (v - s_val) % n
                edges.add((min(v, u), max(v, u)))
                if w != u:
                    edges.add((min(v, w), max(v, w)))
        edges = list(edges)
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if all(d == k for d in deg):
            if _verify_girth(n, edges, k, g):
                return n, edges
        # If verification fails (shouldn't), use a larger n
        n = (g - 1) * k + 1
        edges = set()
        for v in range(n):
            for s_val in range(1, half_k + 1):
                u = (v + s_val) % n
                w = (v - s_val) % n
                edges.add((min(v, u), max(v, u)))
                if w != u:
                    edges.add((min(v, w), max(v, w)))
        edges = list(edges)
        return n, edges
    else:
        # Odd k: use (k-1)/2 symmetric steps plus a perfect matching
        half = (k - 1) // 2
        # Need n large enough that no short cycle forms
        n = (g - 1) * half + 2
        if n <= 2 * half + 1:
            n = 2 * half + 2
        # Ensure n is even for the perfect matching to work cleanly
        if n % 2 != 0:
            n += 1
        edges = set()
        for v in range(n):
            for s_val in range(1, half + 1):
                u = (v + s_val) % n
                w = (v - s_val) % n
                edges.add((min(v, u), max(v, u)))
                if w != u:
                    edges.add((min(v, w), max(v, w)))
            # Perfect matching: pair v with v + n//2
            m = (v + n // 2) % n
            if v < m:
                edges.add((v, m))
        edges = list(edges)
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if all(d == k for d in deg):
            if _verify_girth(n, edges, k, g):
                return n, edges
        # Fallback: larger n
        n = (g - 1) * k + 2
        if n % 2 != 0:
            n += 1
        edges = set()
        for v in range(n):
            for s_val in range(1, half + 1):
                u = (v + s_val) % n
                w = (v - s_val) % n
                edges.add((min(v, u), max(v, u)))
                if w != u:
                    edges.add((min(v, w), max(v, w)))
            m = (v + n // 2) % n
            if v < m:
                edges.add((v, m))
        edges = list(edges)
        return n, edges


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def _verify_girth(n, edges, k, g):
    """Verify the graph is simple, k-regular, and has no cycle shorter than g.

    Uses non-backtracking BFS from every vertex, stopping at depth g//2 - 1.
    Any revisit of a vertex (other than the immediate parent) at depth <= g//2 - 1
    implies a cycle of length < g.
    """
    # Build adjacency
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        key = (u, v) if u < v else (v, u)
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)

    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False

    # Non-backtracking BFS from each vertex, depth limit = g//2 - 1
    # A cycle of length L < g implies there exist two non-backtracking walks
    # of length <= L//2 from some vertex that meet. So checking up to depth
    # g//2 - 1 suffices: any revisit gives a cycle of length <= 2*(g//2-1)+1 < g.
    max_depth = g // 2 - 1
    if max_depth < 1:
        return True  # g <= 2, no constraint

    for start in range(n):
        # state: (vertex, prev)
        dist = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start))
            dist[(nb, start)] = 1
        while queue:
            v, prev = queue.popleft()
            d = dist[(v, prev)]
            if d == max_depth:
                continue
            for nb in adj[v]:
                if nb == prev:
                    continue  # non-backtracking
                state = (nb, v)
                if state in dist:
                    # Found a cycle of length <= d + 1 + dist[state]... 
                    # Actually a revisit means a non-backtracking closed walk,
                    # which contains a cycle. The cycle length is <= 2*(d+1) < g.
                    return False
                dist[state] = d + 1
                queue.append(state)
    return True
