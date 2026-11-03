import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Calculate Moore Bound
    # n >= k * sum_{i=0}^{(g-3)/2} (k-1)^i
    # For odd g
    moore_bound = 0
    term = 1
    for i in range((g - 1) // 2):
        moore_bound += term
        term *= (k - 1)
    moore_bound *= k
    
    # n must be even for k=5 circulant
    if moore_bound % 2 != 0:
        moore_bound += 1
        
    # Search for Circulant Graph
    # k=5 => S = {n/2, s1, -s1, s2, -s2}
    # n must be even
    
    max_n_search = 3000
    if k == 5 and g == 9:
        # Optimized Girth Check for Circulant Graph
        def check_girth_circulant(n, steps, girth_limit):
            dist = [-1] * n
            parent = [-1] * n
            dist[0] = 0
            queue = [0]
            head = 0
            
            while head < len(queue):
                u = queue[head]
                head += 1
                du = dist[u]
                
                if du + 1 >= girth_limit:
                    continue
                
                for s in steps:
                    v = (u + s) % n
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if dist[v] <= du:
                            cycle_len = du + dist[v] + 1
                            if cycle_len < girth_limit:
                                return False
            return True

        attempts_per_n = 50
        for n in range(moore_bound, max_n_search, 2):
            if time.time() > deadline:
                break
            candidates = []
            structured_pairs = [
                (n//5, n//3), (n//4, n//3), (n//6, n//4),
                (n//5, n//4), (n//7, n//3), (n//6, n//5),
                (n//8, n//3), (n//5, n//7), (n//4, n//5),
                (n//6, n//3), (n//7, n//5), (n//8, n//5),
            ]
            for s1, s2 in structured_pairs:
                if 0 < s1 < n//2 and 0 < s2 < n//2 and s1 != s2 and s1 + s2 != n:
                    candidates.append((s1, s2))
            
            for _ in range(attempts_per_n):
                s1 = random.randint(1, n//2 - 1)
                s2 = random.randint(1, n//2 - 1)
                if s1 != s2 and s1 + s2 != n:
                    candidates.append((s1, s2))
                    
            for s1, s2 in candidates:
                if time.time() > deadline:
                    break
                if s1 == s2: continue
                if s1 == 0 or s2 == 0: continue
                if s1 == n//2 or s2 == n//2: continue
                if s1 + s2 == n: continue
                
                steps = [n//2, s1, -s1, s2, -s2]
                
                if check_girth_circulant(n, steps, g):
                    edges = []
                    for i in range(n):
                        for s in steps:
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    # Fallback: Optimized LUW Graph
    return construct_luw_optimized(k, g)

def construct_luw_optimized(k, g):
    q = k
    # LUW graph CD(d, q) has girth 2d + 2
    # We need 2d + 2 >= g
    # d >= (g - 2) / 2
    # d must be integer, so d = ceil((g - 2) / 2)
    # Also d >= 3 for the standard construction validity/efficiency in this context
    d = max(3, (g - 2 + 1) // 2)
    
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
