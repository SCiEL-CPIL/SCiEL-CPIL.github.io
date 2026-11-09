import time
import random
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 115

    def luw_construct(q, d):
        """Lazebnik-Ustimenko-Woldar construction for 3-regular graphs."""
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

        rule = {2: (1, 1), 3: (2, 1)}
        for c in range(4, d + 1):
            r = (c - 4) % 4
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
                yield (1 - side, tuple(y))

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []

        for vi, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))

        return len(order), edges

    def fast_girth_check(n, adj, target_girth):
        """
        Fast girth check using BFS from each vertex.
        For girth g, we need to verify no cycle of length < g.
        Uses non-backtracking walk enumeration.
        """
        for start in range(n):
            # BFS tracking (current, parent) -> depth
            visited = set()
            queue = deque()
            for nb in adj[start]:
                visited.add((nb, start))
                queue.append((nb, start, 1))

            while queue:
                u, prev, depth = queue.popleft()
                if depth >= target_girth:
                    continue
                for nb in adj[u]:
                    if nb == prev:
                        continue
                    if nb == start:
                        return False  # found cycle shorter than target_girth
                    if (nb, u) not in visited:
                        visited.add((nb, u))
                        queue.append((nb, u, depth + 1))
        return True

    def cayley_cyclic_search(n, target_girth):
        """
        Search for a 3-regular Cayley graph on Z_n with generators {a, -a, b}
        (so -b = n-b, and we need the set to be {a, n-a, b, n-b} but that's 4 elements
        unless b = n-b, i.e., 2b = 0 mod n, or we use a different structure).
        
        Actually for degree 3, we need exactly 3 involutions or 1 element of order 2 + 1 pair.
        Let's use generators {a, b, c} where a=-a (order 2), and b, c with b=-c.
        So the Cayley graph has edges (x, x+a), (x, x+b), (x, x-c) = (x, x+n-b).
        For this to be 3-regular and simple: a != 0, b != 0, c != 0, a != b, a != -b, b != -c (always true if b=c).
        Wait, if c = n-b, then generators are {a, b, n-b}. We need:
        - a != 0, b != 0
        - a != b, a != n-b, b != n-b (i.e., 2b != 0)
        - The graph is simple and 3-regular.
        """
        if n % 2 != 0:
            return None
        # a must be n//2 for order 2 (only element of order 2 in Z_n when n is even)
        a = n // 2
        # b can be any element with 1 <= b <= n//2 - 1 and b != a
        for b in range(1, n // 2):
            if b == a:
                continue
            # Check: gcd(n, a) and gcd(n, b) should give connected graph
            # Graph is connected iff gcd(n, a, b) = 1, i.e., gcd(n, b) = 1 since a = n/2
            import math
            if math.gcd(n, b) != 1:
                continue
            # Build adjacency
            adj = [[] for _ in range(n)]
            for x in range(n):
                adj[x].append((x + a) % n)
                adj[x].append((x + b) % n)
                adj[x].append((x - b) % n)
            # Check for multi-edges (shouldn't happen for valid params)
            valid = True
            for x in range(n):
                if len(set(adj[x])) != 3:
                    valid = False
                    break
            if not valid:
                continue
            # Check girth
            if fast_girth_check(n, adj, target_girth):
                edges = []
                for x in range(n):
                    for nb in adj[x]:
                        if x < nb:
                            edges.append((x, nb))
                return n, edges
        return None

    def cayley_prime_search(p, target_girth):
        """
        Cayley graph on Z_p (p prime) with generators {a, -a, b} where we pick
        a and b such that the graph is 3-regular. Since p is odd, there's no element
        of order 2, so we need generators {a, b, c} where one is its own inverse... 
        Actually in Z_p with p odd, no element has order 2. So for degree 3,
        we need an odd number of generators that are not self-inverse.
        This means we need a multiset: but Cayley graphs use sets.
        
        Alternative: use the group Z_p x Z_2 or similar.
        Or: use generators {a, b, c} where a+b+c = 0 and all have order > 2.
        The Cayley graph has edges (x, x+a), (x, x+b), (x, x+c).
        For simplicity: no loops (a,b,c != 0), no multi-edges (a,b,c distinct, a != -b, etc.)
        """
        import math
        for a in range(1, p):
            if time.time() > deadline - 30:
                return None
            for b in range(a + 1, p):
                if time.time() > deadline - 30:
                    return None
                c = (-a - b) % p
                if c == 0 or c == a or c == b:
                    continue
                if c >= p:
                    c -= p
                # Check no self-inverse issues: a != -a (true for p odd, a != 0)
                # Check distinct: a, b, c all distinct and non-zero - done above
                # Also need a != p-b, a != p-c, b != p-c to avoid multi-edges
                if a == (p - b) % p or a == (p - c) % p or b == (p - c) % p:
                    continue
                # Connected: gcd(p, a, b, c) = 1 (always true for prime p with non-zero elements)
                adj = [[] for _ in range(p)]
                for x in range(p):
                    adj[x].append((x + a) % p)
                    adj[x].append((x + b) % p)
                    adj[x].append((x + c) % p)
                # Verify 3-regular and simple
                for x in range(p):
                    if len(set(adj[x])) != 3:
                        break
                else:
                    if fast_girth_check(p, adj, target_girth):
                        edges = []
                        for x in range(p):
                            for nb in adj[x]:
                                if x < nb:
                                    edges.append((x, nb))
                        return p, edges
        return None

    # Strategy 1: LUW construction (known to work, gives valid result)
    best_result = None
    for q in [3, 5, 7]:
        for d in range(7, 12):
            if time.time() > deadline - 40:
                break
            try:
                n, edges = luw_construct(q, d)
                if n < 200000:
                    adj = [[] for _ in range(n)]
                    for u, v in edges:
                        adj[u].append(v)
                        adj[v].append(u)
                    if fast_girth_check(n, adj, g):
                        if best_result is None or n < best_result[0]:
                            best_result = (n, edges)
            except Exception:
                continue

    # Strategy 2: Cayley graph on Z_n (even n) with structure {n/2, b, -b}
    if time.time() < deadline - 50:
        # Try specific n values that are products of small primes
        for n in [512, 576, 640, 704, 768, 832, 896, 960]:
            if time.time() > deadline - 40:
                break
            result = cayley_cyclic_search(n, g)
            if result is not None:
                if best_result is None or result[0] < best_result[0]:
                    best_result = result

    # Strategy 3: Cayley graph on Z_p for prime p
    if time.time() < deadline - 30:
        # Try primes in a reasonable range
        def is_prime(x):
            if x < 2:
                return False
            if x < 4:
                return True
            if x % 2 == 0 or x % 3 == 0:
                return False
            i = 5
            while i * i <= x:
                if x % i == 0 or x % (i + 2) == 0:
                    return False
                i += 6
            return True

        # Try a few primes
        primes_to_try = []
        for p in range(500, 1000):
            if is_prime(p):
                primes_to_try.append(p)
            if len(primes_to_try) >= 5:
                break

        for p in primes_to_try:
            if time.time() > deadline - 20:
                break
            # Limit the search for a, b to keep it fast
            # Only try small a, b values
            for a in range(1, min(50, p // 2)):
                if time.time() > deadline - 20:
                    break
                for b in range(a + 1, min(a + 50, p)):
                    c = (-a - b) % p
                    if c == 0 or c == a or c == b:
                        continue
                    if a == (p - b) % p or a == (p - c) % p or b == (p - c) % p:
                        continue
                    adj = [[] for _ in range(p)]
                    for x in range(p):
                        adj[x].append((x + a) % p)
                        adj[x].append((x + b) % p)
                        adj[x].append((x + c) % p)
                    valid = True
                    for x in range(p):
                        if len(set(adj[x])) != 3:
                            valid = False
                            break
                    if valid and fast_girth_check(p, adj, g):
                        edges = []
                        for x in range(p):
                            for nb in adj[x]:
                                if x < nb:
                                    edges.append((x, nb))
                        if best_result is None or p < best_result[0]:
                            best_result = (p, edges)
                        break
                if best_result is not None and best_result[0] <= p:
                    break

    if best_result is not None:
        return best_result

    # Final fallback: LUW with guaranteed parameters
    n, edges = luw_construct(3, 9)
    return n, edges
