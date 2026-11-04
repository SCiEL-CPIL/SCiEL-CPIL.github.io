import itertools
import time
import random
import math
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    timeout = 110.0  # Safety margin before 120s limit
    
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
        
    def is_prime_power(n):
        """Check if n is a prime power p^e for e >= 1."""
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

    def modinv(a, m):
        """Compute modular inverse of a mod m. Returns None if not invertible."""
        if m <= 1:
            return None
        try:
            return pow(a, -1, m)
        except ValueError:
            return None

    def moore_bound(k, g):
        if k <= 1: return 0
        if g == 1: return 1
        r = g // 2
        if g % 2 == 1:
            # g = 2r + 1
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
    
    def action_on_point(x, a, b, c, d, q):
        if x == q:  # infinity
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

    def check_girth(adj, n, g):
        """
        Returns True if girth >= g.
        Uses BFS from each vertex.
        """
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            
            while queue:
                x = queue.popleft()
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        queue.append(y)
                    elif y != parent[x] and parent[y] != x:
                        # Found a cycle
                        # The cycle length is dist[x] + dist[y] + 1
                        # But we must ensure this is the *shortest* cycle through u?
                        # No, BFS finds shortest paths.
                        # If we encounter a back-edge, the cycle length is dist[x] + dist[y] + 1.
                        # However, this might not be the girth if there's a shorter cycle elsewhere.
                        # But if ANY cycle is < g, we return False.
                        cycle_len = dist[x] + dist[y] + 1
                        if cycle_len < g:
                            return False
        return True

    def build_and_check_pgl(q, k, g, attempts=100):
        """
        Try to build a k-regular graph of girth >= g on q+1 vertices using
        random generators from PGL(2, q).
        """
        n = q + 1
        
        for _ in range(attempts):
            if time.time() - start_time > timeout:
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
            
            # Precompute images
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
            
            # Build adjacency list
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
            
            # Check degree
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
            
            # Check girth
            if check_girth(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        
        return None

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

    # 1. Get the L-U-W baseline
    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_result = (n_luw, edges_luw)
    
    # 2. Try to find a smaller graph using PGL(2, q)
    # We only care about q such that q+1 < best_n
    # Also q+1 >= n_min
    
    q_min = max(2, n_min - 1)
    q_max = best_n - 2  # q+1 < best_n => q < best_n - 1 => q <= best_n - 2
    
    if q_max >= q_min:
        # Iterate q in increasing order to find smallest n first
        # But PGL search is expensive.
        # We will try a limited number of q values.
        # Prioritize q close to n_min? Or just scan?
        # Scanning all q up to best_n-2 might be too many.
        # For k=5, g=9, n_min=426. best_n (L-U-W) might be ~1000.
        # Range 426..998. ~500 values.
        # If we try 500 values * 10 attempts * O(N^2) check, it might be slow.
        
        # Strategy: Try a subset of q values.
        # Prime powers are required.
        # Let's try all prime powers in the range.
        
        # To save time, we can reduce attempts per q if we have many q's.
        # Or just try the first few prime powers.
        
        # Let's try up to 50 prime powers or until timeout.
        count_tried = 0
        for curr_q in range(q_min, q_max + 1):
            if time.time() - start_time > timeout:
                break
            if not is_prime_power(curr_q):
                continue
            
            # Limit attempts per q to ensure we cover more q's
            # If n is small, check is fast. If n is large, check is slow.
            # Let's use 20 attempts per q.
            res = build_and_check_pgl(curr_q, k, g, attempts=20)
            if res:
                n, edges = res
                if n < best_n:
                    best_n = n
                    best_result = res
                    # If we found a much smaller one, we can stop or continue
                    # Since we iterate q increasing, we found the smallest q so far.
                    # But maybe a larger q has a smaller n? No, n=q+1.
                    # So if we find a valid graph for q, it's the best for that q.
                    # We want global min n.
                    # Since we iterate q increasing, the first valid q gives the smallest n.
                    # So we can return immediately?
                    # Yes, because n = q+1. Smaller q means smaller n.
                    return best_result
            
            count_tried += 1
            if count_tried > 100: # Safety cap on number of q's
                break

    return best_result

# EVOLVE-BLOCK-END
