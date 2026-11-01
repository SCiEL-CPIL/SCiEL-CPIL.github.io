import time
import random
import sys
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (i, j) tuples."""
    start_time = time.time()
    deadline = start_time + 115.0

    def find_girth_nb(adj, n, target_girth):
        """Find girth using non-backtracking walk BFS.
        Returns the girth (or a value >= target_girth if girth >= target_girth)."""
        min_cycle = 999
        for s in range(n):
            visited = {}
            queue = deque()
            for v0 in adj[s]:
                visited[(s, v0)] = 0
                queue.append((s, v0, 0))
            while queue:
                u, v, depth = queue.popleft()
                if depth >= target_girth:
                    continue
                for w in adj[v]:
                    if w == u:
                        continue
                    if w == s and depth + 1 >= 3:
                        cl = depth + 1
                        if cl < min_cycle:
                            min_cycle = cl
                            if min_cycle >= target_girth:
                                return min_cycle
                        continue
                    if (v, w) in visited:
                        cl = depth + visited[(v, w)] + 1
                        if cl < min_cycle:
                            min_cycle = cl
                            if min_cycle >= target_girth:
                                return min_cycle
                    else:
                        visited[(v, w)] = depth + 1
                        queue.append((v, w, depth + 1))
            if min_cycle >= target_girth:
                return min_cycle
        return min_cycle

    def seed_construct(k, g):
        """The LUW seed construction."""
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
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
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

    def adj_from_edges(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def is_prime(m):
        if m < 2:
            return False
        if m < 4:
            return True
        if m % 2 == 0 or m % 3 == 0:
            return False
        i = 5
        while i * i <= m:
            if m % i == 0 or m % (i + 2) == 0:
                return False
            i += 6
        return True

    def find_primitive_root(m):
        """Find a primitive root modulo m (for prime m)."""
        if m == 2:
            return 1
        phi = m - 1
        factors = []
        x = phi
        d = 2
        while d * d <= x:
            if x % d == 0:
                factors.append(d)
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            factors.append(x)
        for g in range(2, m):
            ok = True
            for f in factors:
                if pow(g, phi // f, m) == 1:
                    ok = False
                    break
            if ok:
                return g
        return None

    def petersen_edges():
        """Return the 15 edges of the Petersen graph as (u, v) with u < v."""
        edges = []
        for i in range(5):
            edges.append((i, (i + 1) % 5))
        edges.append((5, 7))
        edges.append((7, 9))
        edges.append((9, 6))
        edges.append((6, 8))
        edges.append((8, 5))
        for i in range(5):
            edges.append((i, i + 5))
        return edges

    def k33_edges():
        """Return the 9 edges of K_{3,3}."""
        edges = []
        for i in range(3):
            for j in range(3):
                edges.append((i, j + 3))
        return edges

    def build_voltage_lift(base_edges, n_base, m, voltages):
        """Build a voltage lift. voltages maps (u,v) -> a where u<v.
        Returns (n, adj) or None if not 3-regular."""
        n = n_base * m
        adj = [[] for _ in range(n)]
        for (u, v), a in voltages.items():
            for h in range(m):
                nu = u * m + h
                nv = v * m + ((h + a) % m)
                adj[nu].append(nv)
                adj[nv].append(nu)
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        return n, adj

    # Get the seed construction as fallback
    n_seed, edges_seed = seed_construct(k, g)
    best_n = n_seed
    best_edges = edges_seed

    if k == 3 and g == 16:
        pet_edges = petersen_edges()
        k33_e = k33_edges()

        # Strategy 1: Systematic voltage lifts of Petersen over Z_m (additive)
        # Try primes m from 52 upward; n = 10*m
        for m in range(52, 94):
            if time.time() > deadline - 15:
                break
            if not is_prime(m):
                continue
            root = find_primitive_root(m)
            if root is None:
                continue

            # Try multiple voltage assignments: root^i, root^(i+1), etc.
            for offset in range(10):
                if time.time() > deadline - 15:
                    break
                voltages = {}
                for i, (u, v) in enumerate(pet_edges):
                    voltages[(u, v)] = pow(root, (i + offset) % m, m)
                result = build_voltage_lift(pet_edges, 10, m, voltages)
                if result is None:
                    continue
                n, adj = result
                girth = find_girth_nb(adj, n, g)
                if girth >= g:
                    edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                    break  # found one for this m, try next m

        # Strategy 2: Voltage lifts of K_{3,3} over Z_m; n = 6*m
        # K_{3,3} has girth 4, so we need voltage sums to have high order
        for m in range(86, 156):
            if time.time() > deadline - 10:
                break
            if not is_prime(m):
                continue
            root = find_primitive_root(m)
            if root is None:
                continue

            for offset in range(5):
                if time.time() > deadline - 10:
                    break
                voltages = {}
                for i, (u, v) in enumerate(k33_e):
                    voltages[(u, v)] = pow(root, (i + offset) % m, m)
                result = build_voltage_lift(k33_e, 6, m, voltages)
                if result is None:
                    continue
                n, adj = result
                girth = find_girth_nb(adj, n, g)
                if girth >= g:
                    edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                    break

        # Strategy 3: Randomized voltage search on Petersen for smaller m
        # Use random voltage assignments to find better girth
        for m in range(52, 70):
            if time.time() > deadline - 5:
                break
            if not is_prime(m):
                continue
            for attempt in range(50):
                if time.time() > deadline - 5:
                    break
                voltages = {}
                for i, (u, v) in enumerate(pet_edges):
                    voltages[(u, v)] = random.randint(1, m - 1)
                result = build_voltage_lift(pet_edges, 10, m, voltages)
                if result is None:
                    continue
                n, adj = result
                girth = find_girth_nb(adj, n, g)
                if girth >= g:
                    edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                    break

    # Final verification
    if best_n < 4 or len(best_edges) != best_n * k // 2:
        best_n = n_seed
        best_edges = edges_seed

    return best_n, best_edges
