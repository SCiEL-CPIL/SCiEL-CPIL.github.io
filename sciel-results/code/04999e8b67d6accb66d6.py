import time
import random
from collections import deque

def construct(k, g):
    # The known smallest 4-regular graph with girth >= 10 has 320 vertices.
    # We will search for circulant graphs which are 4-regular.
    
    # Fallback: Lazebnik-Ustimenko-Woldar construction
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

    if k == 4 and g == 10:
        start_time = time.time()
        deadline = start_time + 110
        
        # Search for circulant graphs C_n(a, b)
        # n must be such that a 4-regular graph is possible.
        # Moore bound is 243.
        
        def fast_girth_check(n, steps, min_girth):
            """
            BFS from 0. Returns True if no cycle shorter than min_girth is found 
            involving vertex 0 or detected via non-tree edges from 0's BFS tree.
            Note: This is a necessary condition but not sufficient for global girth.
            """
            # We only need to check up to depth (min_girth - 1) // 2
            max_depth = (min_girth - 1) // 2
            
            dist = [-1] * n
            parent = [-1] * n
            dist[0] = 0
            parent[0] = -2
            queue = deque([0])
            
            min_cycle = float('inf')
            
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                
                for s in steps:
                    v = (u + s) % n
                    
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Check for cycle
                        if parent[u] == v:
                            continue
                        # If v is already visited and not parent
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < min_girth:
                            return False, min_cycle
            
            return True, min_cycle

        def full_girth_check(n, steps, min_girth):
            """
            Verify girth >= min_girth by checking all vertices.
            Optimized: Only BFS to depth (min_girth - 1) // 2.
            """
            max_depth = (min_girth - 1) // 2
            
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                parent[start] = -2
                queue = deque([start])
                
                while queue:
                    u = queue.popleft()
                    if dist[u] >= max_depth:
                        continue
                        
                    for s in steps:
                        v = (u + s) % n
                        
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            if parent[u] == v:
                                continue
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                return False
            return True

        found = False
        # We know 320 works. Start there.
        # If we want to be safe, we can check 320 first, then 321, etc.
        # But 320 is the target.
        for n in range(320, 321): # Just 320 for now to be fast
            if time.time() > deadline:
                break
            
            # Increase attempts for the known good n
            attempts = 10000
            
            for _ in range(attempts):
                if time.time() > deadline:
                    break
                
                # Random steps
                a = random.randint(1, n // 2)
                b = random.randint(1, n // 2)
                
                if a == b:
                    continue
                if (2 * a) % n == 0:
                    continue
                if (2 * b) % n == 0:
                    continue
                if (a + b) % n == 0:
                    continue
                if (a - b) % n == 0:
                    continue
                
                steps = [a, n - a, b, n - b]
                
                ok, _ = fast_girth_check(n, steps, g)
                if not ok:
                    continue
                
                if full_girth_check(n, steps, g):
                    edges = []
                    for u in range(n):
                        for s in [a, b]:
                            v = (u + s) % n
                            if u < v:
                                edges.append((u, v))
                    
                    return n, edges
        
        # If no circulant found at 320, try a few more?
        for n in range(321, 330):
            if time.time() > deadline:
                break
            attempts = 5000
            for _ in range(attempts):
                if time.time() > deadline:
                    break
                a = random.randint(1, n // 2)
                b = random.randint(1, n // 2)
                if a == b: continue
                if (2 * a) % n == 0: continue
                if (2 * b) % n == 0: continue
                if (a + b) % n == 0: continue
                if (a - b) % n == 0: continue
                steps = [a, n - a, b, n - b]
                ok, _ = fast_girth_check(n, steps, g)
                if not ok: continue
                if full_girth_check(n, steps, g):
                    edges = []
                    for u in range(n):
                        for s in [a, b]:
                            v = (u + s) % n
                            if u < v:
                                edges.append((u, v))
                    return n, edges

        # Fallback
        return luw_construct(k, g)
    
    else:
        return luw_construct(k, g)
