import time
import random
import math
from collections import deque, defaultdict

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

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

    def modinv(a, m):
        if m <= 1:
            return None
        try:
            return pow(a, -1, m)
        except (ValueError, ZeroDivisionError):
            return None

    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    # ---------- graph validation ----------
    def check_girth_nb(adj, n, g):
        """Check girth >= g using non-backtracking walks from every vertex."""
        for u in range(n):
            dist = {}
            queue = deque()
            for v in adj[u]:
                dist[(v, u)] = 1
                queue.append((v, u))
            while queue:
                x, prev = queue.popleft()
                d = dist[(x, prev)]
                if d >= g - 1:
                    continue
                for y in adj[x]:
                    if y == prev:
                        continue
                    if y == u:
                        if d + 1 < g:
                            return False
                        continue
                    key = (y, x)
                    if key not in dist:
                        dist[key] = d + 1
                        queue.append((y, x))
        return True

    def is_connected(adj, n):
        if n == 0:
            return True
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
        if not check_girth_nb(adj, n, g):
            return False
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return (n, edges)

    # ---------- Lazebnik-Ustimenko-Woldar baseline ----------
    def luw_construct(k, g):
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
        return len(order), edges

    # ---------- PGL(2, q) machinery ----------
    def action_on_point(x, a, b, c, d, q):
        if x == q:
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
        if c == 0:
            return False
        disc = ((d - a) ** 2 + 4 * b * c) % q
        if disc == 0:
            return False
        return pow(disc, (q - 1) // 2, q) != 1

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
        for i in range(n):
            if len(adj[i]) != k:
                return None
        adj_list = [sorted(adj[i]) for i in range(n)]
        return validate(adj_list, n, k, g)

    def try_structured_pgl(q, k, g):
        if q % 2 == 0 or not is_prime_power(q):
            return None
        n = q + 1
        if n < n_min:
            return None

        def is_nonsquare(a):
            a %= q
            if a == 0:
                return True
            return pow(a, (q - 1) // 2, q) != 1

        cands = []
        for cc in range(1, q):
            if is_nonsquare(1 + 4 * cc):
                cands.append(cc)
            if len(cands) >= 10:
                break
        if len(cands) < 2:
            return None

        h_ok = is_nonsquare(8)
        h2_cands = []
        for cc in range(1, q):
            if is_nonsquare(1 - cc * cc):
                h2_cands.append(cc)
            if len(h2_cands) >= 6:
                break

        # Try multiple combinations of generators
        best_res = None
        for i in range(min(4, len(cands))):
            for j in range(i + 1, min(6, len(cands))):
                c1, c2 = cands[i], cands[j]
                g1 = (1, c1, 1, 0)
                g1i = inv_matrix(*g1, q)
                g2 = (1, c2, 1, 0)
                g2i = inv_matrix(*g2, q)
                if g1i is None or g2i is None:
                    continue
                h = (1, 1, 1, q - 1)

                if h_ok:
                    res = build_from_gens([g1, g1i, g2, g2i, h], q, k, g)
                    if res:
                        if best_res is None or res[0] < best_res[0]:
                            best_res = res
                        return res

                if len(h2_cands) >= 1:
                    for cc in h2_cands[:3]:
                        g3 = (1, q - cc, 1, cc)
                        res = build_from_gens([g1, g1i, g2, g2i, g3], q, k, g)
                        if res:
                            if best_res is None or res[0] < best_res[0]:
                                best_res = res
                            return res
        return best_res

    # ---------- Random PGL search with better candidate generation ----------
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
            out = []
            while len(out) < k:
                m = random_gen()
                if not is_derangement(*m, q):
                    continue
                if random.random() < 0.5:
                    mi = inv_matrix(*m, q)
                    if mi is not None and is_derangement(*mi, q):
                        out.extend([m, mi])
                        if len(out) > k:
                            out = out[:k]
                else:
                    out.append(m)
            return out

        seen = set()
        best_local = None
        while time.time() < budget:
            gens = random_gens()
            key = tuple(gens)
            if key in seen:
                continue
            seen.add(key)
            res = build_from_gens(gens, q, k, g)
            if res:
                if best_local is None or res[0] < best_local[0]:
                    best_local = res
                return res
        return best_local

    # ---------- Benson construction via unitriangular groups ----------
    def gf_mul_table(q):
        """Build multiplication table for GF(q) where q is a prime power."""
        if is_prime(q):
            return [[(a * b) % q for b in range(q)] for a in range(q)]
        return None

    def benson_construct(k, g):
        best = None
        for q in range(max(2, k - 2), k + 5):
            if not is_prime_power(q):
                continue
            n = q * q + q + 1
            if n < n_min:
                continue
            if best is not None and n >= best[0]:
                continue

            if k != q + 1 and k != q:
                continue

            # Skip Benson for now - it typically gives even girth
            break

        return best

    # ---------- Local improvement: try to reduce graph size ----------
    def try_reduce_graph(n, edges, k, g, budget):
        """Try to find a smaller subgraph or a different construction with fewer vertices."""
        if n <= n_min:
            return None

        # Try slightly smaller q values for PGL
        for q_try in range(max(3, n - 5), n - 1):
            if time.time() > budget:
                break
            if not is_prime_power(q_try):
                continue
            res = try_structured_pgl(q_try, k, g)
            if res and res[0] < n:
                return res

        return None

    # ---------- Main ----------
    n_luw, edges_luw = luw_construct(k, g)
    best_n = [n_luw]
    best_result = (n_luw, edges_luw)

    # Try Benson construction
    benson_res = benson_construct(k, g)
    if benson_res and benson_res[0] < best_n[0]:
        best_n[0] = benson_res[0]
        best_result = benson_res

    # Try structured PGL for small q - expanded range
    q_max_struct = min(best_n[0] - 2, 120)
    for curr_q in range(max(3, n_min - 1), q_max_struct + 1):
        if time.time() > deadline - 40.0:
            break
        res = try_structured_pgl(curr_q, k, g)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
            # Don't return immediately - keep searching for even better
            if best_n[0] <= n_min + 5:
                break

    # Random PGL search with improved budget allocation
    q_cursor = max(3, n_min - 1)
    while q_cursor + 1 < best_n[0] and time.time() < deadline - 15.0:
        # Use more time per q value for better search
        budget = min(deadline - 15.0, time.time() + 3.0)
        res = random_search_pgl(q_cursor, k, g, budget)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
        q_cursor += 1

    # Final attempt: try to reduce the best graph found
    if time.time() < deadline - 5.0:
        reduced = try_reduce_graph(best_n[0], [], k, g, deadline - 5.0)
        if reduced and reduced[0] < best_n[0]:
            best_n[0] = reduced[0]
            best_result = reduced

    return best_result

# EVOLVE-BLOCK-END
