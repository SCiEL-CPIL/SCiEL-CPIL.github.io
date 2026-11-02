import math
import time
from collections import deque

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    # Moore bound calculation
    D = (g - 2) // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(D))
    
    # Search for circulant graph C_n(a, b)
    # Girth check function
    def has_girth_at_least(n, a, b, g):
        # BFS on (v, prev)
        # Start from 0
        # Neighbors of 0 are a, -a, b, -b
        # We want to find the shortest cycle back to 0
        
        # dist[(v, prev)] = steps
        dist = {}
        q = deque()
        
        # Initial neighbors
        start_neighbors = [a % n, (-a) % n, b % n, (-b) % n]
        # Remove duplicates
        start_neighbors = list(set(start_neighbors))
        
        for sn in start_neighbors:
            dist[(sn, 0)] = 1
            q.append((sn, 0, 1))
            
        while q:
            v, prev, d = q.popleft()
            # If we are at 0, we found a cycle
            # But we start from neighbors of 0, so v != 0 initially.
            # We check if next step is 0.
            
            # Neighbors of v are v+a, v-a, v+b, v-b
            for step in [a, -a, b, -b]:
                w = (v + step) % n
                if w == prev:
                    continue
                if w == 0:
                    # Cycle length is d + 1
                    if d + 1 < g:
                        return False
                    else:
                        # This cycle is valid, but we need to ensure no shorter one exists.
                        # Since BFS, this is the shortest cycle through 0 found so far.
                        # If this is the first time we hit 0, it is the girth.
                        # We can return True immediately?
                        # No, there might be a shorter cycle not found yet?
                        # BFS guarantees shortest path.
                        # So the first time we hit 0 is the girth.
                        return True
                if (w, v) not in dist:
                    dist[(w, v)] = d + 1
                    q.append((w, v, d + 1))
        return True # No cycle found? Graph is a tree? Impossible for regular graph.

    deadline = time.time() + 110
    
    # Try small n first
    # n must be even? Not necessarily, but for 4-regular, n must be even (sum of degrees = 2E).
    # 4n = 2E => 2n = E. n can be odd? No, 4n is even, 2E is even.
    # But a 4-regular graph must have even number of vertices?
    # Sum of degrees = 4n. This is even.
    # But if n is odd, can we have a 4-regular graph?
    # Yes, e.g. K5 is 4-regular.
    # But for girth 10, n must be large.
    
    # Limit search to reasonable n
    max_n = min(moore_bound + 100, 500) # Search up to 500
    
    for n in range(moore_bound, max_n + 1):
        if time.time() > deadline:
            break
        # a, b generators
        # a < b < n/2
        for a in range(1, n // 2 + 1):
            if math.gcd(a, n) != 1:
                continue
            for b in range(a + 1, n // 2 + 1):
                if math.gcd(b, n) != 1:
                    continue
                if math.gcd(a - b, n) != 1 and math.gcd(a + b, n) != 1:
                    # If gcd(a-b, n) > 1 or gcd(a+b, n) > 1, graph might be disconnected or have small cycles?
                    # Actually, we just need connectedness: gcd(a, b, n) = 1.
                    # Since gcd(a, n)=1 and gcd(b, n)=1, gcd(a, b, n) is always 1.
                    pass
                
                if has_girth_at_least(n, a, b, g):
                    # Construct edges
                    edges = []
                    for i in range(n):
                        edges.append((i, (i + a) % n))
                        edges.append((i, (i + b) % n))
                    return n, edges
    # Fallback
    return luw_construct(k, g)
