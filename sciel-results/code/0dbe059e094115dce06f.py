import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Moore bound for k=5, g=9 is 428.
    # We search for n >= 428.
    # Since we use circulant graph on Z_n, n must be even for k=5 (odd degree).
    # Start n from 430 (next even >= 428).
    
    best_n = None
    best_a = None
    best_b = None
    
    # Fallback to LUW if search fails
    def luw_fallback():
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

    def check_girth(n, a, b):
        # Steps: n/2, a, -a, b, -b
        steps = [n // 2, a, -a, b, -b]
        # Normalize steps to 0..n-1
        steps = [s % n for s in steps]
        # Remove duplicates
        steps = list(set(steps))
        if len(steps) != 5:
            return False # Not 5-regular or has duplicates
        
        # Neighbors of 0
        neigh_0 = set(steps)
        
        # Non-backtracking BFS
        # State: (node, prev_node)
        # Start: (0, -1)
        # Depth 1: (s, 0) for s in steps
        
        visited = set()
        queue = []
        for s in steps:
            queue.append((s, 0, 1))
            visited.add((s, 0))
            
        # We want to check if we hit a neighbor of 0 at depth < 8
        # Cycle length = depth + 1.
        # We want cycle length >= 9 => depth >= 8.
        # So if we hit neigh_0 at depth d where 2 <= d <= 7, it's bad.
        
        while queue:
            node, prev, depth = queue.pop(0)
            
            # Check if node is in neigh_0
            if node in neigh_0:
                # If depth >= 2, we have a cycle of length depth + 1
                if depth < 8:
                    return False
                # If depth >= 8, it's a valid cycle (or longer), but we only care about short ones.
                # Actually, if we hit it at depth 8, cycle is 9, which is OK.
                # But we stop checking short ones.
                continue
            
            # Expand
            next_depth = depth + 1
            if next_depth > 7:
                continue
                
            for s in steps:
                new_node = (node + s) % n
                new_prev = node
                # Avoid backtracking
                if new_node == prev:
                    continue
                
                state = (new_node, new_prev)
                if state not in visited:
                    visited.add(state)
                    queue.append((new_node, new_prev, next_depth))
                    
        return True

    # Search
    # n must be even.
    n = 430 
    while time.time() < deadline:
        if n % 2 != 0:
            n += 1
            continue
            
        # Try some random a, b
        # a, b must be distinct from 0, n/2 and each other (mod n)
        # and a != -b
        found = False
        for _ in range(20): # Try 20 random pairs per n
            a = random.randint(1, n - 1)
            b = random.randint(1, n - 1)
            
            # Check distinctness
            if a == b or a == -b % n: continue
            if a == n // 2 or b == n // 2: continue
            if a == 0 or b == 0: continue
            
            if check_girth(n, a, b):
                # Construct graph
                edges = []
                for i in range(n):
                    for s in [n // 2, a, -a, b, -b]:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
        
        n += 2
        
    # Fallback
    return luw_fallback()
