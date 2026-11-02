"""Portfolio SEARCH: try multiple explicit families, keep best (smallest) valid graph."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys
from collections import deque

def construct(k, g):
    deadline = time.time() + 115.0  # safety margin before 150s kill
    rng = random.Random(12345)

    def check_deadline():
        return time.time() > deadline

    # --- Utility: validate graph (simple, k-regular, girth >= g) ---
    def validate(n, edges, k, g):
        if n <= 0:
            return False
        adj = [[] for _ in range(n)]
        seen = set()
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return False
            key = (min(u,v), max(u,v))
            if key in seen:
                return False
            seen.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            if len(adj[u]) != k:
                return False
        # Girth check via non-backtracking BFS from each vertex.
        # A graph has girth >= g iff there is no non-backtracking closed walk of length < g.
        # Equivalently: for every vertex s, BFS in the "non-backtracking" sense up to depth g-2
        # must not return to s.
        # We do: for each start s, do a BFS where state is (current, prev) to avoid backtracking.
        # Track shortest distance to each (v, prev) state. If we reach (s, any) at distance < g, cycle exists.
        # To keep it fast, we cap depth at g-1.
        for s in range(n):
            # dist[(v, prev)] = shortest non-backtracking walk length from s to v arriving from prev
            dist = {}
            q = deque()
            for nb in adj[s]:
                dist[(nb, s)] = 1
                q.append((nb, s))
            while q:
                u, prev = q.popleft()
                d = dist[(u, prev)]
                if d >= g - 1:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue  # non-backtracking
                    nd = d + 1
                    if v == s:
                        # closed non-backtracking walk of length nd
                        if nd < g:
                            return False
                        continue
                    key = (v, u)
                    if key not in dist or dist[key] > nd:
                        dist[key] = nd
                        q.append((v, u))
        return True

    def try_graph(n, edges, k, g):
        if not validate(n, edges, k, g):
            return None
        return n, edges

    best = None
    best_n = float('inf')

    def consider(n, edges):
        nonlocal best, best_n
        if n < best_n:
            result = try_graph(n, edges, k, g)
            if result is not None:
                best = result
                best_n = n
        if check_deadline():
            raise TimeoutError

    # --- Family 1: Lazebnik-Ustimenko-Woldar (from seed) ---
    def luw_family(q=None, d=None):
        if q is None:
            q = k
        if d is None:
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,)+(0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # --- Family 2: Circulant / folded circulant constructions ---
    def circulant_family():
        # Try circulant graphs: vertices 0..n-1, edges (i, i+s) mod n for a set of shifts S.
        # Degree = 2*|S| (if no shift is self-inverse) or less. For odd n, each shift s != n/2 gives degree 2.
        # For even n, shift n/2 gives degree 1.
        # We need degree k and girth >= g.
        # For k even: k = 2*m, use m shifts. For k odd: harder, use circulant + one extra.
        # This is a heuristic; we'll try a few small n and shift sets.
        results = []
        # Moore bound lower estimate: n >= 1 + k * sum_{i=0}^{floor((g-1)/2)-1} (k-1)^i
        # For g odd, g=2r+1: n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
        # For g even, g=2r: n >= k * sum_{i=0}^{r-1} (k-1)^i
        if g % 2 == 1:
            r = (g-1)//2
            moore = 1 + k * sum((k-1)**i for i in range(r))
        else:
            r = g//2
            moore = k * sum((k-1)**i for i in range(r))
        # Try n from moore to moore + 200 (small range)
        max_n = moore + 300
        if k % 2 == 0:
            m = k // 2
            for n in range(max(moore, 2*m+1), max_n+1):
                if n % 2 == 0 and m == 1:
                    # shift n/2 gives degree 1, not 2. Skip.
                    pass
                # Try shift set {1, 2, ..., m} (simplest circulant)
                shifts = list(range(1, m+1))
                edges = []
                for i in range(n):
                    for s in shifts:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                results.append((n, edges))
        else:
            # k odd: use circulant with m = (k-1)//2 shifts (degree 2*m = k-1) plus one more edge per vertex
            # This is tricky; skip for now or use a different construction.
            pass
        return results

    # --- Family 3: Incidence graphs of projective planes (if applicable) ---
    # For k = q+1 where q is prime power, the incidence graph of PG(2,q) is (q+1)-regular with girth 6.
    # Not useful for g=9, but include for completeness.
    def incidence_family():
        # q must be prime power, k = q+1, girth = 6. Only useful if g <= 6.
        if g > 6:
            return None
        # Find q such that q+1 = k
        q = k - 1
        # Check q is prime power
        def is_prime_power(q):
            if q < 2:
                return False
            for p in range(2, q+1):
                if p * p > q:
                    break
                if q % p == 0:
                    # check q = p^a
                    temp = q
                    while temp % p == 0:
                        temp //= p
                    if temp == 1:
                        return True
            return False
        if not is_prime_power(q):
            return None
        # Build incidence graph of PG(2,q): points and lines
        # Number of points = number of lines = q^2 + q + 1
        # Total vertices = 2*(q^2+q+1)
        # This is complex to build explicitly; skip for now.
        return None

    # --- Run all families ---
    # Run LUW with default parameters
    try:
        n, e = luw_family()
        consider(n, e)
    except Exception as ex:
        pass

    # Try LUW with different d values
    for d_try in range(max(3, g-6), g+2):
        if check_deadline():
            break
        try:
            n, e = luw_family(d=d_try)
            consider(n, e)
        except Exception as ex:
            pass

    # Try circulant family
    try:
        circ_results = circulant_family()
        for n, e in circ_results:
            if check_deadline():
                break
            consider(n, e)
    except Exception as ex:
        pass

    if check_deadline():
        if best is not None:
            return best
        # Fallback: return LUW
        n, e = luw_family()
        return n, e

    if best is not None:
        return best

    # Fallback: return LUW
    n, e = luw_family()
    return n, e

# EVOLVE-BLOCK-END
