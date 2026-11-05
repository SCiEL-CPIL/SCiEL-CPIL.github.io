import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    def remaining():
        return deadline - time.time()

    # ---------- arithmetic helpers ----------
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

    def is_prime_power(n):
        if n < 2:
            return False
        if is_prime(n):
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % p == 0:
                t = n
                while t % p == 0:
                    t //= p
                return t == 1
        return False

    def is_quadratic_residue(a, q):
        """Check if a is a quadratic residue mod q (works for prime powers)."""
        a = a % q
        if a == 0:
            return True
        if not is_prime_power(q):
            return True  # conservative: assume residue
        # Find prime base p of q
        p = None
        for cand in range(2, int(math.isqrt(q)) + 1):
            if q % cand == 0:
                t = q
                while t % cand == 0:
                    t //= cand
                if t == 1:
                    p = cand
                    break
        if p is None:
            p = q  # q is prime
        # Reduce to mod p
        a_p = a % p
        if a_p == 0:
            return True
        # Euler's criterion mod p (p is prime)
        return pow(a_p, (p - 1) // 2, p) == 1

    def modinv(a, m):
        """Extended Euclidean algorithm for modular inverse (works for any modulus)."""
        if m <= 1:
            return None
        a = a % m
        if a == 0:
            return None
        # Try Python's built-in first (fast for primes)
        try:
            return pow(a, -1, m)
        except ValueError:
            pass
        # Extended Euclidean fallback
        old_r, r = a, m
        old_s, s = 1, 0
        while r != 0:
            q = old_r // r
            old_r, r = r, old_r - q * r
            old_s, s = s, old_s - q * s
        if old_r != 1:
            return None
        return old_s % m

    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:  # g = 2r + 1
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:  # g = 2r
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    # ---------- graph validation ----------
    def check_girth_fast(adj, n, g):
        """
        Check girth using BFS from each vertex.
        Optimized: Stop BFS as soon as we find a cycle shorter than g.
        Uses non-backtracking walk certification for exact girth check.
        """
        half_g = g // 2
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                x = queue.popleft()
                if dist[x] >= half_g:
                    continue  # No need to expand further
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        if dist[y] < half_g:
                            queue.append(y)
                    else:
                        # Check for cycle: edge (x,y) where both visited
                        if y != parent[x] and parent[y] != x:
                            cycle_len = dist[x] + dist[y] + 1
                            if cycle_len < g:
                                return False
        return True

    def is_connected(adj, n):
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return count == n

    def validate(adj, n, k, g):
        for i in range(n):
            if len(adj[i]) != k:
                return False
        if not is_connected(adj, n):
            return False
        if not check_girth_fast(adj, n, g):
            return False
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return (n, edges)

    # ---------- Lazebnik-Ustimenko-Woldar baseline (deterministic) ----------
    def luw_construct(k, g):
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

    # ---------- PGL(2, q) machinery (q odd prime power) ----------
    def action_on_point(x, a, b, c, d, q):
        if x == q:  # infinity
            if c == 0:
                return q
            inv_c = modinv(c, q)
            if inv_c is None:
                return None
            return (a * inv_c) % q
        num = (a * x + b) % q
        den = (c * x + d) % q
        if den == 0:
            return q
        inv_den = modinv(den, q)
        if inv_den is None:
            return None
        return (num * inv_den) % q

    def inv_matrix(a, b, c, d, q):
        det = (a * d - b * c) % q
        det_inv = modinv(det, q)
        if det_inv is None:
            return None
        return (d * det_inv % q, -b * det_inv % q, -c * det_inv % q, a * det_inv % q)

    def is_derangement(a, b, c, d, q):
        """Check if Möbius transform has no fixed points on P^1(F_q)."""
        a, b, c, d = a % q, b % q, c % q, d % q
        # infinity fixed iff c == 0
        if c == 0:
            return False
        # finite fixed points: c x^2 + (d - a) x - b = 0
        # discriminant = (d-a)^2 + 4bc
        disc = ((d - a) ** 2 + 4 * b * c) % q
        if disc == 0:
            return False  # one finite fixed point
        # Check if disc is a quadratic residue mod q
        if is_quadratic_residue(disc, q):
            return False  # two finite fixed points
        return True

    def build_from_gens(gens, q, k, g):
        n = q + 1
        adj = [set() for _ in range(n)]
        for i in range(n):
            x = i if i < q else q
            for gen in gens:
                y = action_on_point(x, gen[0], gen[1], gen[2], gen[3], q)
                if y is None or y == i:
                    return None
                j = y if y < q else q
                adj[i].add(j)
                adj[j].add(i)
        # Quick degree check before full validation
        for i in range(n):
            if len(adj[i]) != k:
                return None
        return validate(adj, n, k, g)

    # ---------- main ----------
    # 1. deterministic baseline
    n_luw, edges_luw = luw_construct(k, g)
    best_n = [n_luw]
    best_result = (n_luw, edges_luw)

    # 2. Random search in PGL(2, q) for q < best_n
    # We prioritize smaller q values as they are more likely to have high girth
    # relative to their size, and we want to find n < best_n.
    
    def random_search_pgl(q, k, g, budget):
        n = q + 1
        if n < n_min or n >= best_n[0]:
            return None
        if q % 2 == 0 or not is_prime_power(q):
            return None

        def random_gen():
            while True:
                a = random.randrange(q)
                b = random.randrange(q)
                c = random.randrange(q)
                d = random.randrange(q)
                if (a * d - b * c) % q != 0:
                    return (a, b, c, d)

        def random_gens():
            """Generate k generators, preferring involutions (order 2) for better girth."""
            out = []
            attempts = 0
            while len(out) < k and attempts < k * 50:
                attempts += 1
                m = random_gen()
                mi = inv_matrix(*m, q)
                if mi is None:
                    continue
                # Check derangement
                if not is_derangement(*m, q):
                    continue
                # Prefer involutions: m^2 = identity
                a2 = (m[0]*m[0] + m[1]*m[2]) % q
                b2 = (m[1]*(m[0]+m[3])) % q
                c2 = (m[2]*(m[0]+m[3])) % q
                d2 = (m[3]*m[3] + m[1]*m[2]) % q
                is_involution = (a2 == 1 and b2 == 0 and c2 == 0 and d2 == 1)
                
                if is_involution or random.random() < 0.5:
                    # Use both m and m^-1 (they're distinct for non-involutions)
                    out.append(m)
                    if mi != m and len(out) < k:
                        out.append(mi)
            # Fill remaining with random derangements
            while len(out) < k:
                m = random_gen()
                if is_derangement(*m, q):
                    out.append(m)
            return out[:k]

        seen = set()
        while time.time() < budget:
            gens = random_gens()
            key = tuple(gens)
            if key in seen:
                continue
            seen.add(key)
            if not all(is_derangement(*gen, q) for gen in gens):
                continue
            res = build_from_gens(gens, q, k, g)
            if res:
                return res
        return None

    # Search q values
    q_cursor = max(3, n_min - 1)
    # We want to find n < best_n[0], so q < best_n[0] - 1
    max_q = best_n[0] - 1
    
    while q_cursor < max_q and time.time() < deadline - 5.0:
        # Allocate time based on q
        # Smaller q are more promising
        time_left = deadline - 5.0 - time.time()
        if time_left <= 0:
            break
        
        # Give more time to smaller q
        # q range is [n_min, max_q]
        # Let's try a few q values with increasing time budget
        # Or just sweep q with a fixed small budget?
        # Better: Try q, if fails, move to q+1.
        
        # Budget for this q: proportional to 1/q? Or fixed?
        # Let's try 1-2 seconds per q value for the first few, then less.
        budget = 1.0 if q_cursor < n_min + 50 else 0.5
        
        res = random_search_pgl(q_cursor, k, g, budget)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
            # Found improvement, can try to find even smaller?
            # But smaller q means smaller n. We are iterating q upwards.
            # So if we found one at q, smaller q (already tried) failed.
            # Larger q gives larger n.
            # So we should return?
            # Wait, we iterate q from low to high.
            # If we find a valid graph at q, it has n=q+1.
            # Any larger q will have larger n.
            # So we can return immediately.
            return best_result
        
        q_cursor += 1

    return best_result

# EVOLVE-BLOCK-END
