import time
from collections import deque
from itertools import product

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def luw_construct(k, g, d_override=None, rule_override=None):
        q = k
        if d_override is not None:
            d = d_override
        else:
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
        if rule_override is not None:
            rule = rule_override
        else:
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

    def verify_girth(n, edges, target_girth):
        adj = [[] for _ in range(n)]
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            if u < 0 or v < 0 or u >= n or v >= n:
                return False
            key = (u, v) if u < v else (v, u)
            if key in seen:
                return False
            seen.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            if len(adj[u]) != k:
                return False
        half = (target_girth - 1) // 2
        for s in range(n):
            dist = {}
            queue = deque()
            for nb in adj[s]:
                st = (nb, s)
                dist[st] = 1
                queue.append(st)
            while queue:
                v, p = queue.popleft()
                dcur = dist[(v, p)]
                if dcur >= half:
                    continue
                for w in adj[v]:
                    if w == p:
                        continue
                    st = (w, v)
                    if st in dist:
                        cyc = dcur + dist[st] + 1
                        if cyc < target_girth:
                            return False
                    else:
                        dist[st] = dcur + 1
                        queue.append(st)
        return True

    def gen_rules(d, variant=0):
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            if variant == 0:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            elif variant == 1:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 1), (c0, 1), (c0 + 1, 1), (1, c0)][r]
            elif variant == 2:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(2, c0 - 1), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            elif variant == 3:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][(r + 1) % 4]
            elif variant == 4:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][(r + 2) % 4]
            elif variant == 5:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][(r + 3) % 4]
            elif variant == 6:
                if c % 2 == 1:
                    rule[c] = (1, c - 2)
                else:
                    rule[c] = (c - 1, 1)
            elif variant == 7:
                if c % 3 == 0:
                    rule[c] = (1, c - 2)
                elif c % 3 == 1:
                    rule[c] = (c - 1, 1)
                else:
                    rule[c] = (c, 1)
            else:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return rule

    if k != 3 or g != 16:
        return luw_construct(k, g)

    best_n = float('inf')
    best_edges = None

    # ===== STRUCTURAL CHANGE: Cubic Cayley graph search on finite groups =====
    # A cubic Cayley graph Cay(G, {a, a^{-1}, b}) where a*b = b*a (abelian) 
    # or more generally, we search over small groups.
    # For abelian groups Z_m x Z_n, the Cayley graph with generators {±e1, ±e2, ±e3}
    # is 6-regular. For cubic, we need |S|=3 with S closed under inverse.
    # So S = {a, a^{-1}, b} where b = b^{-1} (order 2) or S = {a, a^{-1}, b, b^{-1}} with |S|=4...
    # Actually for cubic: S must have exactly 3 elements, closed under inverse.
    # Options: S = {a, a^{-1}, c} where c = c^{-1} (c has order 2), or
    # S = {a, b, c} where a=a^{-1}, b=b^{-1}, c=c^{-1} (all order 2).
    
    def cayley_abelian(m, n, a1, a2, b1, b2):
        """Cubic Cayley graph on Z_m x Z_n with generators ±(a1,a2) and (b1,b2) where (b1,b2)=-(b1,b2) mod (m,n)."""
        # (b1,b2) must be self-inverse: 2*b1 ≡ 0 mod m, 2*b2 ≡ 0 mod n
        if (2 * b1) % m != 0 or (2 * b2) % n != 0:
            return None
        # Also need (a1,a2) != (b1,b2) and (a1,a2) != -(b1,b2) and a != 0
        if (a1, a2) == (0, 0):
            return None
        if (a1, a2) == (b1, b2) or (a1, a2) == ((-b1) % m, (-b2) % n):
            return None
        # Check that ±(a1,a2) are distinct and different from (b1,b2)
        neg_a = ((-a1) % m, (-a2) % n)
        if neg_a == (a1, a2):
            # a is self-inverse, so we'd have only 2 distinct neighbors
            # Need to handle: if a=-a, then S={a, b} gives degree 2, not 3
            # Skip this case for cubic
            return None
        # Build edges
        edges = set()
        for i in range(m):
            for j in range(n):
                v = i * n + j
                # neighbor via +a
                ni = (i + a1) % m
                nj = (j + a2) % n
                w = ni * n + nj
                if v < w:
                    edges.add((v, w))
                # neighbor via -a
                ni = (i - a1) % m
                nj = (j - a2) % n
                w = ni * n + nj
                if v < w:
                    edges.add((v, w))
                # neighbor via b (self-inverse)
                ni = (i + b1) % m
                nj = (j + b2) % n
                w = ni * n + nj
                if v < w:
                    edges.add((v, w))
        return m * n, list(edges)

    def cayley_cyclic(m, a, b):
        """Cubic Cayley graph on Z_m with generators ±a and b (b self-inverse)."""
        if (2 * b) % m != 0:
            return None
        if a % m == 0:
            return None
        if (a % m) == (b % m) or (a % m) == ((-b) % m):
            return None
        if (2 * a) % m == 0:
            return None  # a self-inverse, only 2 distinct neighbors
        edges = set()
        for i in range(m):
            # +a
            w = (i + a) % m
            if i < w:
                edges.add((i, w))
            # -a
            w = (i - a) % m
            if i < w:
                edges.add((i, w))
            # b
            w = (i + b) % m
            if i < w:
                edges.add((i, w))
        return m, list(edges)

    # Search cyclic groups first (faster)
    # For girth >= 16, need m large enough. Moore bound for (3,16) is 512.
    # Try m from ~80 to ~300 (even, for self-inverse b)
    for m in range(80, 320, 2):
        if time.time() > deadline:
            break
        # b must satisfy 2b ≡ 0 mod m, so b = m/2
        b = m // 2
        for a in range(1, m // 2 + 1):
            if (2 * a) % m == 0:
                continue
            if a == b or a == m - b:
                continue
            result = cayley_cyclic(m, a, b)
            if result is None:
                continue
            n, edges = result
            if n >= best_n:
                continue
            if len(edges) != m * 3 // 2:
                continue
            if verify_girth(n, edges, g):
                best_n = n
                best_edges = edges
                break  # found valid for this m

    # Search Z_m x Z_n groups
    for m in range(10, 50):
        if time.time() > deadline:
            break
        for n in range(10, 50):
            if m * n >= best_n:
                continue
            if m * n < 80:
                continue
            # b = (b1, b2) self-inverse: 2*b1 ≡ 0 mod m, 2*b2 ≡ 0 mod n
            b1s = [0, m // 2] if m % 2 == 0 else [0]
            b2s = [0, n // 2] if n % 2 == 0 else [0]
            for b1 in b1s:
                for b2 in b2s:
                    if (b1, b2) == (0, 0):
                        continue
                    for a1 in range(0, m):
                        for a2 in range(0, n):
                            if (a1, a2) == (0, 0):
                                continue
                            if (2 * a1) % m == 0 and (2 * a2) % n == 0:
                                continue  # a self-inverse
                            result = cayley_abelian(m, n, a1, a2, b1, b2)
                            if result is None:
                                continue
                            nn, edges = result
                            if nn >= best_n:
                                continue
                            if verify_girth(nn, edges, g):
                                best_n = nn
                                best_edges = edges
                                break
                        if best_n < m * n:
                            break
                    if best_n < m * n:
                        break
                if best_n < m * n:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Also try LUW as fallback
    for d in [7, 8, 9, 10, 11, 12]:
        if time.time() > deadline:
            break
        max_variants = 12 if d == 7 else (8 if d == 8 else 5)
        for variant in range(max_variants):
            if time.time() > deadline:
                break
            rule = gen_rules(d, variant)
            n, edges = luw_construct(k, g, d_override=d, rule_override=rule)
            if n >= best_n:
                continue
            if verify_girth(n, edges, g):
                best_n = n
                best_edges = edges

    if best_edges is not None:
        return best_n, best_edges

    return luw_construct(k, g)
