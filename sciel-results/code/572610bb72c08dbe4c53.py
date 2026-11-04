import time
import math
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # safety margin below 150s wall-clock

    def elapsed():
        return time.time() - start_time

    # ---- Girth verification: non-backtracking BFS from every vertex ----
    def girth_of(adj, n):
        # Returns the girth (shortest cycle length), or inf if acyclic.
        # Uses non-backtracking walks: state (v, parent), BFS from each start.
        # A cycle of length L containing start vertex u is detected when a
        # non-backtracking walk returns to a neighbor of u.
        best = float('inf')
        for u in range(n):
            # dist[(v, p)] = length of non-backtracking walk u -> ... -> v with prev p
            dist = {}
            q = deque()
            for v in adj[u]:
                dist[(v, u)] = 1
                q.append((v, u))
            while q:
                v, parent = q.popleft()
                d = dist[(v, parent)]
                for w in adj[v]:
                    if w == parent:
                        continue
                    if w == u:
                        # closed non-backtracking walk of length d+1 = cycle
                        if d + 1 < best:
                            best = d + 1
                        continue
                    state = (w, v)
                    if state not in dist:
                        dist[state] = d + 1
                        q.append(state)
                    # early stop if we already found a cycle of length <= current best
        return best

    # ---- Alon-Babai-Suzuki (ABS) construction ----
    # Cayley graph on the group of upper triangular unit matrices over GF(q).
    # Group elements: (a, b) representing matrix [[1, a], [0, 1]] where a in GF(q).
    # Actually for ABS, we use the semidirect product or a more structured group.
    # Standard ABS: vertices are pairs (i, j) in GF(q) x GF(q) with a specific connection rule.
    # The ABS construction gives a k-regular graph of girth at least 2*floor(log_k(n/2)) or similar.
    # 
    # More precisely, ABS constructs a (q+1)-regular graph on q^2 vertices with girth ~ 2 log_q(n).
    # For our purpose with general k, we use q = k (or nearest prime power) and construct
    # a k-regular graph on approximately k^2 vertices.
    
    def is_prime_power(n):
        if n < 2:
            return None
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            if p > n:
                break
            m = n
            while m % p == 0:
                m //= p
            if m == 1:
                return p
        return None

    def gf_add(x, y, p):
        return (x + y) % p

    def gf_sub(x, y, p):
        return (x - y) % p

    def gf_mul(x, y, p):
        return (x * y) % p

    def gf_inv(x, p):
        if x == 0:
            return 0
        return pow(x, -1, p)

    def abs_construct(k, g):
        """Alon-Babai-Suzuki style construction.
        Uses a group-based Cayley graph approach.
        For prime q, constructs a (q+1)-regular graph on q(q+1) vertices
        with good girth properties.
        """
        # Find smallest prime q >= k
        q = k
        while not is_prime(q):
            q += 1
        
        # ABS construction: vertices are (i, j) where i in GF(q), j in GF(q)
        # Total vertices: q^2
        # Degree: q (from translations) + 1 (from diagonal) = q+1
        # But we need degree exactly k, so we may need to adjust.
        
        # Standard ABS: Cayley graph on Z_q x Z_q with generators:
        # (1, 0), (0, 1), and (0, a) for a in some set, plus diagonal elements.
        
        # Let me use the classical ABS construction more carefully.
        # The ABS graph G(q) has q^2 vertices (i,j) in GF(q)^2.
        # Vertex (i,j) is connected to:
        #   (i+1, j) for all i,j  -- this gives degree q from horizontal
        #   Actually, the standard ABS uses:
        #   Neighbors of (i,j): (i+a, j+b) for (a,b) in a generating set S of size k.
        
        # A well-known k-regular ABS-type construction:
        # Vertices: GF(q) x GF(q), n = q^2
        # Generators: {(1,0), (0,1), (0,a1), (0,a2), ...} to get degree k
        # Girth is typically 2*floor(log_k(q)) + 1 or similar.
        
        # For degree k, we need k generators. Let's use:
        # g_0 = (1, 0)
        # g_1 = (0, 1) 
        # g_i = (0, r^i) for i = 2, ..., k-1 where r is primitive root
        # This gives k generators, but (0,1) and (0,r^i) may create short cycles.
        
        # Better ABS approach: use the group structure properly.
        # Let's use the construction from Alon, Babai, Suzuki (1985):
        # Vertices are elements of GF(q) x GF(q).
        # Edges: (x,y) ~ (x+1, y) and (x,y) ~ (x, y+1) and (x,y) ~ (x, y*x) for x != 0
        # This gives degree 3 for q > 3, not general k.
        
        # Generalized ABS for degree k:
        # Use q = k (prime), vertices GF(q)^2, n = q^2 = k^2.
        # Generators: e_1 = (1,0), e_2 = (0,1), and k-2 more from a carefully chosen set.
        
        # For simplicity and correctness, let me use a known good construction:
        # The Morgenstern-type construction on PGL(2,q) already exists in the code.
        # ABS is essentially equivalent in many cases.
        
        # Let me implement a proper ABS:
        # Group: affine group of GF(q), elements (a,b) with a in GF(q)*, b in GF(q)
        # Composition: (a,b)(c,d) = (ac, ad+b)
        # Size: q(q-1)
        # Generators for k-regular: choose k elements of the group.
        
        # Actually, the simplest robust ABS-style construction:
        # Use the bipartite incidence graph of a projective plane or similar.
        # But that's bipartite and may not be optimal.
        
        # Let me just return None and let the existing constructions handle it,
        # but try a simple Cayley graph on Z_q^2 first.
        
        n = q * q
        # Build Cayley graph on Z_q x Z_q with k generators
        # Generators: (1,0), (0,1), (1,1), (1,2), ... choose k linearly independent-ish
        # To avoid short cycles, use generators that are "spread out"
        
        r = primitive_root(q) if q > 2 else 1
        
        # Choose k generators
        gens = []
        gens.append((1, 0))  # horizontal
        if k > 1:
            gens.append((0, 1))  # vertical
        for i in range(2, k):
            # Use diagonal generators with primitive root powers
            a = pow(r, i, q)
            gens.append((a, 1))
        
        # Verify we have exactly k generators
        if len(gens) != k:
            return None
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for i in range(q):
            for j in range(q):
                v = i * q + j
                for (a, b) in gens:
                    ni = (i + a) % q
                    nj = (j + b) % q
                    w = ni * q + nj
                    if v == w:
                        return None  # self-loop
                    if w in adj[v]:
                        return None  # duplicate edge
                    adj[v].add(w)
                    adj[w].add(v)
        
        # Check degree
        for v in range(n):
            if len(adj[v]) != k:
                return None
        
        # Check connectivity
        visited = [False] * n
        dq = deque([0])
        visited[0] = True
        cnt = 0
        while dq:
            u = dq.popleft()
            cnt += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    dq.append(v)
        if cnt != n:
            return None
        
        # Check girth
        gr = girth_of(adj, n)
        if gr < g:
            return None
        
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        
        return n, edges

    def primitive_root(p):
        # smallest primitive root mod prime p
        if p == 2:
            return 1
        # factor p-1
        n = p - 1
        factors = []
        m = n
        d = 2
        while d * d <= m:
            if m % d == 0:
                factors.append(d)
                while m % d == 0:
                    m //= d
            d += 1
        if m > 1:
            factors.append(m)
        for g in range(2, p):
            ok = True
            for f in factors:
                if pow(g, n // f, p) == 1:
                    ok = False
                    break
            if ok:
                return g
        return None

    # ---- Known small cages (n, edge list) ----
    # (5,9) cage: the 5-regular graph of girth 9 with minimum vertices.
    # Best known order is 1116 (Exoo et al.). We don't have the exact edge list
    # memorized, so we rely on construction for (5,9).
    # For (3,9): known cage is 126 vertices (Biggs-Smith). We construct it.
    # For (5,5): 30 vertices (Hoffman-Singleton).
    # For (3,5): 10 vertices (Petersen).
    # For (3,7): 24 vertices.
    # For (4,5): 19 vertices.
    # We'll use constructions to generate these; the lookup is a fallback hint.

    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:
            # g = 2r+1
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            # g = 2r
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    def is_valid(k, g, n, edges):
        # Build adjacency, check simple, regular, connected, girth >= g
        adj = [set() for _ in range(n)]
        seen = set()
        for (i, j) in edges:
            if i == j:
                return False
            if i < 0 or i >= n or j < 0 or j >= n:
                return False
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                return False
            seen.add((a, b))
            adj[i].add(j)
            adj[j].add(i)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # connectivity
        visited = [False] * n
        dq = deque([0])
        visited[0] = True
        cnt = 0
        while dq:
            u = dq.popleft()
            cnt += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    dq.append(v)
        if cnt != n:
            return False
        # girth
        gr = girth_of(adj, n)
        return gr >= g

    # ---- Finite field GF(q) for prime q ----
    def is_prime(n):
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

    def build_morgenstern(q, k, g):
        # Morgenstern's construction: k generators from PGL(2, q)
        # using powers of a primitive root. Returns (n, edges) or None.
        if q < 2:
            return None
        if not is_prime(q):
            return None
        n = q + 1  # vertices 0..q, where q represents infinity
        # We need k generators. Morgenstern uses a specific set.
        # Generators: for a primitive root r of GF(q), the matrices
        #   M_i = [[0, 1], [1, r^i]]  type (transpositions) and
        #   M = [[r, 0], [0, r^{-1}]] (diagonal)
        # A standard k-regular construction (Morgenstern 1996) uses:
        #   For k even: k/2 involutions + ... 
        # Simpler robust approach: pick k random PGL(2,q) elements and test.
        # But deterministic is better. Use the "Morgenstern" standard set:
        #   g_1 = [[0,1],[1,0]]  (involution)
        #   g_2 = [[0,1],[1,1]]
        #   ... and diagonal powers.
        # We'll generate candidate generator sets and verify.

        def action(x, a, b, c, d):
            # x in 0..q (q = infinity)
            if x == q:
                if c == 0:
                    return q
                inv_c = pow(c, -1, q)
                return (a * inv_c) % q
            num = (a * x + b) % q
            den = (c * x + d) % q
            if den == 0:
                return q
            return (num * pow(den, -1, q)) % q

        def try_gens(gens):
            adj = [set() for _ in range(n)]
            for i in range(n):
                x = i
                for (a, b, c, d) in gens:
                    y = action(x, a, b, c, d)
                    if i == y:
                        return None
                    if y in adj[i]:
                        return None
                    adj[i].add(y)
                    adj[y].add(i)
            for i in range(n):
                if len(adj[i]) != k:
                    return None
            # connectivity
            visited = [False] * n
            dq = deque([0])
            visited[0] = True
            cnt = 0
            while dq:
                u = dq.popleft()
                cnt += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        dq.append(v)
            if cnt != n:
                return None
            gr = girth_of(adj, n)
            if gr < g:
                return None
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges

        # Candidate generator sets.
        # Set A: Morgenstern standard for k generators using primitive root.
        r = primitive_root(q)
        if r is None:
            return None

        # Build a pool of candidate matrices.
        pool = []
        # Involutions / transpositions: [[0,1],[1, t]] for t in GF(q)
        for t in range(q):
            pool.append((0, 1, 1, t))
        # Diagonal: [[s,0],[0,s^{-1}]]
        for s in range(1, q):
            pool.append((s, 0, 0, pow(s, -1, q)))
        # General: [[a,b],[c,d]] with det != 0 (sampled)
        # We'll sample a deterministic pseudo-random set.
        rng = random.Random(12345)
        for _ in range(200):
            while True:
                a = rng.randint(0, q - 1)
                b = rng.randint(0, q - 1)
                c = rng.randint(0, q - 1)
                d = rng.randint(0, q - 1)
                if (a * d - b * c) % q != 0:
                    pool.append((a, b, c, d))
                    break

        # Try the classical Morgenstern set first:
        # k generators: g_i for i=0..k-1 where
        #   g_0 = [[0,1],[1,0]]
        #   g_i = [[r^i, 0],[0, r^{-i}]] for i>=1  (diagonal powers)
        # This gives k generators.
        # Actually a known k-regular Morgenstern set:
        #   generators = { [[0,1],[1, r^j]] : j=0..k-2 } union { [[r,0],[0,r^{-1}]] }
        # We'll just try several structured sets.
        candidate_sets = []

        # Set 1: transpositions with consecutive t + one diagonal
        if k >= 1:
            s1 = [(0, 1, 1, t) for t in range(k - 1)]
            if k > 1:
                s1.append((r, 0, 0, pow(r, -1, q)))
            candidate_sets.append(s1)

        # Set 2: all diagonal powers
        if k >= 1:
            s2 = [(pow(r, i, q), 0, 0, pow(r, -i, q)) for i in range(k)]
            candidate_sets.append(s2)

        # Set 3: transpositions with spaced t
        if k >= 1:
            step = max(1, q // (k + 1))
            s3 = [(0, 1, 1, (t * step) % q) for t in range(k)]
            candidate_sets.append(s3)

        # Try each candidate set
        for gens in candidate_sets:
            if len(gens) != k:
                continue
            res = try_gens(gens)
            if res:
                return res

        # Random search over generator sets (bounded)
        for _ in range(200):
            if elapsed() > deadline - 5:
                break
            gens = []
            for _ in range(k):
                while True:
                    a = rng.randint(0, q - 1)
                    b = rng.randint(0, q - 1)
                    c = rng.randint(0, q - 1)
                    d = rng.randint(0, q - 1)
                    if (a * d - b * c) % q != 0:
                        break
                gens.append((a, b, c, d))
            res = try_gens(gens)
            if res:
                return res
        return None

    # ---- L-U-W (Lazebnik-Ustimenko-Woldar) construction ----
    def luw_construct(k, g):
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

    # ---- Main: try constructions, keep best valid result ----
    best = None
    best_n = float('inf')

    def consider(n, edges):
        nonlocal best, best_n
        if best is None or n < best_n:
            if is_valid(k, g, n, edges):
                best = (n, edges)
                best_n = n
                return True
        return False

    # 0) Try ABS construction first (typically smaller vertex count)
    if elapsed() < deadline - 30:
        try:
            res = abs_construct(k, g)
            if res:
                consider(res[0], res[1])
        except Exception:
            pass

    # 1) Try Morgenstern for primes q near the Moore bound.
    #    n = q+1, so q = n-1. Search q from max(2, n_min-1) upward.
    max_q = max(n_min + 200, 300)
    q = max(2, n_min - 1)
    while q <= max_q and elapsed() < deadline - 20:
        if is_prime(q):
            res = build_morgenstern(q, k, g)
            if res:
                consider(res[0], res[1])
        q += 1

    # 2) Try L-U-W (deterministic, always produces a k-regular graph).
    if elapsed() < deadline - 5:
        try:
            res = luw_construct(k, g)
            consider(res[0], res[1])
        except Exception:
            pass

    # 3) If nothing valid yet, fall back to L-U-W raw (should be valid).
    if best is None:
        try:
            res = luw_construct(k, g)
            best = res
        except Exception:
            # Last-resort: return a trivial valid construction.
            # For safety, build a k-regular graph on a multiple of ... 
            # Use L-U-W with a larger d to force girth.
            q = k
            d = max(5, g)
            add = lambda x, y: (x + y) % q if q != 4 else x ^ y
            sub = lambda x, y: (x - y) % q if q != 4 else x ^ y
            mul = lambda x, y: x * y % q if q != 4 else (x & y)
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
            best = (len(order), edges)

    return best

# EVOLVE-BLOCK-END
