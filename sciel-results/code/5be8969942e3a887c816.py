import time
import math
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = 140.0

    def elapsed():
        return time.time() - start_time

    def remaining():
        return deadline - start_time

    # ---- Girth verification: non-backtracking BFS from every vertex ----
    def girth_of(adj, n):
        best = float('inf')
        for u in range(n):
            dist = {}
            q = deque()
            for v in adj[u]:
                dist[(v, u)] = 1
                q.append((v, u))
            while q:
                v, parent = q.popleft()
                d = dist[(v, parent)]
                if d >= best:
                    continue
                for w in adj[v]:
                    if w == parent:
                        continue
                    if w == u:
                        if d + 1 < best:
                            best = d + 1
                        continue
                    state = (w, v)
                    if state not in dist:
                        dist[state] = d + 1
                        q.append(state)
        return best

    def girth_of_fast(adj, n, lower_bound):
        if lower_bound <= 2:
            return 2
        max_depth = lower_bound - 1
        for u in range(n):
            visited = [bytearray([0]) * n for _ in range(n)]
            q = deque()
            for v in adj[u]:
                visited[u][v] = 1
                q.append((v, u))
            while q:
                v, parent = q.popleft()
                d = visited[parent][v]
                if d >= max_depth:
                    continue
                for w in adj[v]:
                    if w == parent:
                        continue
                    if w == u:
                        return d + 1
                    if visited[v][w] == 0:
                        visited[v][w] = d + 1
                        q.append((w, v))
        return lower_bound

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

    def primitive_root(p):
        if p == 2:
            return 1
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

    def is_valid(k, g, n, edges):
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
        gr = girth_of(adj, n)
        return gr >= g

    # ---- L-U-W construction ----
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

    # ---- PGL(2,q) Cayley graph with systematic generator search ----
    def build_pgl_cayley(q, k, g, max_attempts=500):
        """Build a k-regular Cayley graph on PGL(2,q) acting on P^1(q) (n=q+1 points)."""
        if q < 2 or not is_prime(q):
            return None
        n = q + 1
        if n < 2 * k:
            return None

        inv = [0] * (q + 1)
        for i in range(1, q):
            inv[i] = pow(i, q - 2, q)
        inv[q] = 0

        def action(x, a, b, c, d):
            if x == q:
                if c == 0:
                    return q
                return (a * inv[c]) % q
            num = (a * x + b) % q
            den = (c * x + d) % q
            if den == 0:
                return q
            return (num * inv[den]) % q

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
            gr = girth_of_fast(adj, n, g)
            if gr < g:
                return None
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges

        involutions = []
        for b in range(q):
            for c in range(q):
                if (b * c + 1) % q != 0:
                    involutions.append((1, b, c, q - 1))
        for b in range(1, q):
            for c in range(1, q):
                involutions.append((0, b, c, 0))

        if len(involutions) < k:
            return None

        rng = random.Random(42 + q)
        n_inv = len(involutions)
        step = max(1, n_inv // k)
        structured = [involutions[i * step % n_inv] for i in range(k)]
        res = try_gens(structured)
        if res:
            return res

        for _ in range(max_attempts):
            if elapsed() > deadline - 15:
                break
            gens = rng.sample(involutions, k)
            res = try_gens(gens)
            if res:
                return res
        return None

    # ---- Local search: edge swap to improve girth ----
    def local_search_improve(k, g, n, edges, max_iterations=2000):
        """Try edge swaps to eliminate short cycles."""
        adj = [set() for _ in range(n)]
        for (i, j) in edges:
            adj[i].add(j)
            adj[j].add(i)

        edge_set = set()
        for (i, j) in edges:
            a, b = (i, j) if i < j else (j, i)
            edge_set.add((a, b))

        def find_short_cycle(adj, n, g):
            """Find a cycle of length < g, or None."""
            for u in range(n):
                dist = {}
                q = deque()
                for v in adj[u]:
                    dist[(v, u)] = 1
                    q.append((v, u))
                while q:
                    v, parent = q.popleft()
                    d = dist[(v, parent)]
                    if d >= g:
                        continue
                    for w in adj[v]:
                        if w == parent:
                            continue
                        if w == u:
                            return d + 1
                        state = (w, v)
                        if state not in dist:
                            dist[state] = d + 1
                            q.append(state)
            return None

        rng = random.Random(123)
        for _ in range(max_iterations):
            if elapsed() > deadline - 20:
                break
            cycle_len = find_short_cycle(adj, n, g)
            if cycle_len is None:
                break

            # Pick a random edge and try to swap
            i = rng.randrange(n)
            neighbors = list(adj[i])
            if len(neighbors) < 2:
                continue
            j = rng.choice(neighbors)
            # Find a non-neighbor of j to swap with
            non_neighbors = [v for v in range(n) if v != i and v != j and v not in adj[j]]
            if not non_neighbors:
                continue
            w = rng.choice(non_neighbors)
            if w in adj[i]:
                continue

            # Try swap: remove (i,j), add (i,w)
            adj[i].remove(j)
            adj[j].remove(i)
            adj[i].add(w)
            adj[w].add(i)
            edge_set.discard((min(i, j), max(i, j)))
            edge_set.add((min(i, w), max(i, w)))

            # Check if this helped
            new_cycle = find_short_cycle(adj, n, g)
            if new_cycle is not None and new_cycle <= cycle_len:
                # Revert
                adj[i].remove(w)
                adj[w].remove(i)
                adj[i].add(j)
                adj[j].add(i)
                edge_set.discard((min(i, w), max(i, w)))
                edge_set.add((min(i, j), max(i, j)))

        edges = list(edge_set)
        return n, edges

    # ---- Main ----
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

    # 1) L-U-W baseline
    if elapsed() < deadline - 30:
        try:
            res = luw_construct(k, g)
            if res:
                consider(res[0], res[1])
        except Exception:
            pass

    # 2) Try PGL(2,q) for various primes
    n_min = moore_bound(k, g)
    q_start = max(3, n_min - 1)
    q_limit = max(q_start + 300, 500)
    q = q_start
    while q <= q_limit and elapsed() < deadline - 25:
        if is_prime(q):
            try:
                res = build_pgl_cayley(q, k, g, max_attempts=300)
                if res:
                    consider(res[0], res[1])
            except Exception:
                pass
        q += 1

    # 3) Local search on best result
    if best is not None and elapsed() < deadline - 25:
        try:
            n, edges = best
            res = local_search_improve(k, g, n, edges, max_iterations=3000)
            if res:
                consider(res[0], res[1])
        except Exception:
            pass

    # 4) Fallback
    if best is None:
        try:
            res = luw_construct(k, g)
            best = res
        except Exception:
            if k == 1 and g == 1:
                best = (1, [])
            elif k == 2:
                n = max(g, 3)
                edges = [(i, (i + 1) % n) for i in range(n)]
                best = (n, edges)
            else:
                n = k + 1
                edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
                best = (n, edges)

    return best

# EVOLVE-BLOCK-END
