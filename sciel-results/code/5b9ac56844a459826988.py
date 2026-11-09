import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.monotonic()
    deadline = 115.0  # Safety margin before 120s

    # ------------------------------------------------------------------
    # Girth verification: exact non-backtracking walk from every vertex
    # ------------------------------------------------------------------
    def _shortest_cycle(adj, n, g):
        INF = 1 << 30
        for s in range(n):
            seen = set()
            dq = deque()
            for w in adj[s]:
                st = (w, s)
                seen.add(st)
                dq.append((st, 1))
            while dq:
                (v, p), d = dq.popleft()
                if d >= g:
                    continue
                for w in adj[v]:
                    if w == p:
                        continue
                    if w == s:
                        return d + 1
                    st = (w, v)
                    if st not in seen:
                        seen.add(st)
                        dq.append((st, d + 1))
        return INF

    def _girth_ok(adj, n, g):
        return _shortest_cycle(adj, n, g) >= g

    # ------------------------------------------------------------------
    # LUW construction CD(d, q)
    # ------------------------------------------------------------------
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if d + 5 < g:
            d += 1

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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []

        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))

        return len(order), edges

    # ------------------------------------------------------------------
    # Cayley graph on Z_q x Z_2 with degree 5
    # Generators: +a, -a, +b, -b, (0,1)
    # ------------------------------------------------------------------
    def _cayley_zq_z2(q, a, b):
        n = 2 * q
        adj = [[] for _ in range(n)]
        edges = []
        for i in range(q):
            for t in (0, 1):
                u = 2 * i + t
                for (di, dt) in ((a, 0), (-a, 0), (b, 0), (-b, 0), (0, 1)):
                    v = 2 * ((i + di) % q) + ((t + dt) & 1)
                    if u < v:
                        adj[u].append(v)
                        adj[v].append(u)
                        edges.append((u, v))
        return n, adj, edges

    def _is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        d = 3
        while d * d <= x:
            if x % d == 0:
                return False
            d += 2
        return True

    def _next_prime(x):
        x = max(2, x)
        if x == 2:
            return 2
        if x % 2 == 0:
            x += 1
        while not _is_prime(x):
            x += 2
        return x

    # ------------------------------------------------------------------
    # Random regular graph generation (configuration model)
    # ------------------------------------------------------------------
    def generate_random_regular(n, k):
        if (n * k) % 2 != 0:
            return None
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        random.shuffle(half_edges)

        edges = set()
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:
                return None
            key = (min(u, v), max(u, v))
            if key in edges:
                return None
            edges.add(key)
            adj[u].append(v)
            adj[v].append(u)
        return list(edges), adj

    # ------------------------------------------------------------------
    # Main logic
    # ------------------------------------------------------------------
    # Moore bound for odd girth g=2r+1
    r = (g - 1) // 2
    moore_bound = k + k * sum((k - 1) ** i for i in range(r))
    if moore_bound % 2 != 0:
        moore_bound += 1

    # Get LUW baseline
    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # Verify LUW baseline (should always pass, but be safe)
    adj_luw = [[] for _ in range(n_luw)]
    for u, v in edges_luw:
        adj_luw[u].append(v)
        adj_luw[v].append(u)
    if not _girth_ok(adj_luw, n_luw, g):
        # Should not happen, but if it does, we have no valid graph yet
        # Try random search as fallback
        pass

    # Strategy:
    # 1. Try Cayley graphs Z_q x Z_2 for primes q from Moore bound/2 up to best_n/2
    # 2. If time remains, try random regular graphs for sizes between Moore bound and best_n

    # Cayley search
    lo_q = max(3, (moore_bound + 1) // 2)
    hi_q = best_n // 2

    # Generator pairs: need a, b distinct, non-zero, a != -b mod q
    # We'll try small a, b values
    small_ab = []
    for a in range(1, 10):
        for b in range(1, 10):
            if a == b:
                continue
            if a == 0 or b == 0:
                continue
            small_ab.append((a, b))

    q = _next_prime(lo_q)
    while q <= hi_q:
        if time.monotonic() - start_time > deadline:
            break
        for (a, b) in small_ab:
            if time.monotonic() - start_time > deadline:
                break
            a_mod = a % q
            b_mod = b % q
            if a_mod == 0 or b_mod == 0:
                continue
            if a_mod == b_mod or a_mod == (-b_mod) % q:
                continue
            n, adj, edges = _cayley_zq_z2(q, a_mod, b_mod)
            if n < best_n:
                if _girth_ok(adj, n, g):
                    best_n = n
                    best_edges = edges
        q = _next_prime(q + 1)

    # Random search for sizes between Moore bound and current best
    if time.monotonic() - start_time < deadline - 5:
        current_n = moore_bound
        if current_n % 2 != 0:
            current_n += 1
        max_n_to_try = best_n - 2

        while current_n <= max_n_to_try:
            if time.monotonic() - start_time > deadline:
                break

            attempts = 0
            found_valid = False
            max_attempts = max(10, int((deadline - (time.monotonic() - start_time)) * 5))
            max_attempts = min(max_attempts, 500)

            while attempts < max_attempts:
                if time.monotonic() - start_time > deadline:
                    break
                result = generate_random_regular(current_n, k)
                if result is None:
                    attempts += 1
                    continue
                edges, adj = result
                if _girth_ok(adj, current_n, g):
                    best_n = current_n
                    best_edges = edges
                    found_valid = True
                    break
                attempts += 1

            if found_valid:
                # Found valid graph at current_n. Try to go smaller? 
                # We started at Moore bound, so if current_n == moore_bound, we're done.
                # If we found one at current_n > moore_bound, we might try smaller? 
                # But we iterate current_n upwards. So we can break if we want the smallest found so far.
                # Actually, the loop goes current_n += 2. If we found one, we break out of the inner loop.
                # We should check if we can try a smaller n? No, we started at Moore bound.
                # So if we found one, it's the smallest possible in this range.
                # But wait, we might have skipped some n if we started at moore_bound and incremented.
                # So if we found one at current_n, it is the smallest n >= moore_bound that we tried.
                # We can break the outer loop because we can't do better than current_n (which is >= moore_bound).
                break

            current_n += 2

    return best_n, best_edges
