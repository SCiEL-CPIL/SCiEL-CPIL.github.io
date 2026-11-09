import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140

    def check_girth_circulant(n, S, target_girth):
        # S is a list of 3 generators.
        # Check if shortest non-backtracking closed walk from 0 has length >= target_girth.
        
        # BFS on (vertex, prev_step_val)
        # prev_step_val is the value of the step taken to arrive at vertex.
        # None for start.
        
        visited = {}
        queue = deque()
        queue.append((0, None, 0))
        visited[(0, None)] = 0
        
        while queue:
            v, prev_val, d = queue.popleft()
            
            if d >= target_girth:
                continue
                
            for s in S:
                # Check backtracking
                if prev_val is not None and s == (-prev_val) % n:
                    continue
                    
                new_v = (v + s) % n
                
                # If we return to 0
                if new_v == 0 and d > 0:
                    # Cycle length is d + 1
                    return d + 1
                
                state = (new_v, s)
                if state not in visited:
                    visited[state] = d + 1
                    queue.append((new_v, s, d + 1))
                    
        return float('inf')

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
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
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

    luw_n, luw_edges = luw_construct(k, g)
    
    result_n = luw_n
    result_edges = luw_edges
    
    # Search for circulant graph
    # n from 512 to luw_n - 1
    # Use fixed seed for reproducibility? No, random is fine, but deterministic is better for debugging.
    # However, we want to find *a* solution.
    
    # To ensure we don't miss small n, we iterate n.
    # We can increase trials if n is small.
    
    for n in range(512, luw_n):
        if time.time() > deadline - 20:
            break
            
        # Number of trials
        # More trials for smaller n?
        trials = 100
        
        for _ in range(trials):
            candidates = list(range(1, n))
            if n % 2 == 0:
                # Exclude n/2
                if n // 2 in candidates:
                    candidates.remove(n // 2)
            
            if len(candidates) < 3:
                continue
                
            S = random.sample(candidates, 3)
            
            # Check S and -S disjoint
            neg_S = [(-s) % n for s in S]
            if len(set(S + neg_S)) != 6:
                continue
                
            girth = check_girth_circulant(n, S, g)
            
            if girth >= g:
                edges = []
                for i in range(n):
                    for s in S:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                edges = list(set(edges))
                
                result_n = n
                result_edges = edges
                return result_n, result_edges
                
    return result_n, result_edges
