import time
import random
import math
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    timeout = 115.0  # Safety margin before 120s limit

    def time_check():
        return time.time() - start_time > timeout

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
                if t == 1:
                    return True
                return False
        return False

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

    def check_girth_nonbacktracking(adj, n, g):
        """
        Check girth >= g using non-backtracking walks from every vertex.
        Uses BFS with parent tracking to avoid backtracking.
        """
        # For girth >= g, we need to verify no cycle of length < g
        # Using non-backtracking BFS: from each vertex, explore up to depth (g-1)//2
        # and check for collisions
        
        max_depth = (g - 1) // 2
        
        for u in range(n):
            # BFS from u, tracking (node, parent) to avoid backtracking
            # dist[node] = shortest non-backtracking distance from u
            visited = {}  # (node, parent) -> distance
            queue = deque()
            for v in adj[u]:
                queue.append((v, u, 1))
                visited[(v, u)] = 1
            
            while queue:
                node, parent, dist = queue.popleft()
                if dist >= max_depth:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue  # non-backtracking
                    if (nb, node) in visited:
                        # Found a cycle
                        # The cycle length is dist + visited[(nb,node)] + 1
                        # Actually, we need to be more careful
                        # If we reached nb from node at distance dist,
                        # and nb was previously reached from node at distance d2,
                        # then we have a cycle of length dist + d2 (if the paths diverge)
                        # This is complex; let's use a simpler check
                        pass
                    else:
                        visited[(nb, node)] = dist + 1
                        queue.append((nb, node, dist + 1))
        
        # Simpler approach: use BFS from each vertex with parent tracking
        # to detect cycles of length < g
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            
            while queue:
                x = queue.popleft()
                dx = dist[x]
                if dx >= max_depth:
                    continue
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dx + 1
                        parent[y] = x
                        queue.append(y)
                    elif y != parent[x]:
                        # Potential cycle: x-y where y is already visited
                        # and y is not the parent of x
                        cycle_len = dist[x] + dist[y] + 1
                        if cycle_len < g:
                            return False
        return True

    def luw_construct(q, d):
        """
        Lazebnik-Ustimenko-Woldar graph CD(d, q).
        Returns (n, edges) for the connected component of the origin.
        """
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

    def try_luw(q, d, k, g):
        """Try LUW construction with given q and d, return result if valid."""
        n, edges = luw_construct(q, d)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return None
        
        # Check girth
        if check_girth_nonbacktracking(adj, n, g):
            return n, edges
        
        return None

    def modinv(a, m):
        if m <= 1:
            return None
        try:
            return pow(a, -1, m)
        except ValueError:
            return None

    def action_on_point(x, a, b, c, d, q):
        if x == q:
            if c == 0:
                return q
            else:
                inv_c = modinv(c, q)
                if inv_c is None:
                    return None
                return (a * inv_c) % q
        else:
            num = (a * x + b) % q
            den = (c * x + d) % q
            if den == 0:
                return q
            else:
                inv_den = modinv(den, q)
                if inv_den is None:
                    return None
                return (num * inv_den) % q

    def build_and_check_pgl(q, k, g, attempts=50):
        n = q + 1
        
        for _ in range(attempts):
            if time_check():
                return None
            
            gens = []
            valid = True
            for _ in range(k):
                while True:
                    a = random.randint(0, q - 1)
                    b = random.randint(0, q - 1)
                    c = random.randint(0, q - 1)
                    d = random.randint(0, q - 1)
                    det = (a * d - b * c) % q
                    if det != 0:
                        break
                gens.append((a, b, c, d))
            
            images = [[None] * k for _ in range(n)]
            valid = True
            for i in range(n):
                x = i if i < q else q
                for j in range(k):
                    img = action_on_point(x, gens[j][0], gens[j][1], gens[j][2], gens[j][3], q)
                    if img is None:
                        valid = False
                        break
                    images[i][j] = img
                if not valid:
                    break
            
            if not valid:
                continue
            
            adj = [set() for _ in range(n)]
            is_simple = True
            
            for i in range(n):
                for j in range(k):
                    img = images[i][j]
                    if img == i:
                        is_simple = False
                        break
                    adj[i].add(img)
                    adj[img].add(i)
                if not is_simple:
                    break
            
            if not is_simple:
                continue
            
            for i in range(n):
                if len(adj[i]) != k:
                    is_simple = False
                    break
            
            if not is_simple:
                continue
            
            # Check connectivity
            visited = [False] * n
            queue = deque([0])
            visited[0] = True
            count = 0
            while queue:
                u = queue.popleft()
                count += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
            if count != n:
                continue
            
            if check_girth_nonbacktracking(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        
        return None

    best_n = float('inf')
    best_result = None

    # Strategy 1: Try LUW with various q values
    # LUW CD(d, q) is q-regular with girth >= d+5 for odd d
    # So we need d >= g-5 and q = k
    # But we can also try other q values if the construction works
    
    # For k=5, g=9: need d >= 4, but d must be odd for the girth guarantee
    # So d >= 5. Try d = 5, 7, 9, ...
    # q should be a prime power >= k
    
    d_min = max(3, g - 5)
    if d_min % 2 == 0:
        d_min += 1  # Ensure odd for girth guarantee
    
    # Try q = k first (most natural)
    q_candidates = [k]
    # Add a few other prime powers
    for qp in range(k + 1, k + 50):
        if is_prime_power(qp):
            q_candidates.append(qp)
            if len(q_candidates) >= 10:
                break
    
    for q in q_candidates:
        if time_check():
            break
        
        for d in range(d_min, d_min + 6, 2):  # Try odd d values
            if time_check():
                break
            
            result = try_luw(q, d, k, g)
            if result:
                n, edges = result
                if n < best_n:
                    best_n = n
                    best_result = result
    
    # Strategy 2: Try PGL(2,q) for smaller q values
    if best_n == float('inf') or best_n > n_min * 3:
        q_min = max(2, n_min - 1)
        q_max = min(best_n - 2 if best_n != float('inf') else q_min + 100, q_min + 200)
        
        count_tried = 0
        for curr_q in range(q_min, q_max + 1):
            if time_check():
                break
            if not is_prime_power(curr_q):
                continue
            
            res = build_and_check_pgl(curr_q, k, g, attempts=30)
            if res:
                n, edges = res
                if n < best_n:
                    best_n = n
                    best_result = res
                    return best_result
            
            count_tried += 1
            if count_tried > 50:
                break

    if best_result is None:
        # Fallback: use LUW with default parameters
        d = d_min
        best_n, best_result = luw_construct(k, d)
    
    return best_result

# EVOLVE-BLOCK-END
