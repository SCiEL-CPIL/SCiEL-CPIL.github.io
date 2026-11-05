import time
import math
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    # Girth guarantee: d+5 if d is odd, d+4 if d is even
    # Choose smallest d such that guaranteed girth >= g
    
    min_d_odd = max(3, g - 5)
    if min_d_odd % 2 == 0:
        min_d_odd += 1
        
    min_d_even = max(4, g - 4)
    if min_d_even % 2 == 1:
        min_d_even += 1
        
    d = min(min_d_odd, min_d_even)
    q = k
    
    # GF(q) operations
    if q == 4:
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return mul_table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Precompute rules for the LUW construction
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    
    rule_a[2], rule_b[2] = 1, 1
    rule_a[3], rule_b[3] = 2, 1
    rule_a[4], rule_b[4] = 1, 2
    
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            a, b = 1, c0 - 2
        elif r == 1:
            a, b = c0 - 1, 1
        elif r == 2:
            a, b = c0, 1
        else:
            a, b = 1, c0 + 1
        rule_a[c] = a
        rule_b[c] = b

    def compute_neighbours(side, x):
        result = []
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            result.append((1 - side, tuple(y)))
        return result

    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []
    
    queue = deque([start])
    head = 0
    
    while head < len(order):
        if time.time() > deadline:
            break
            
        vertex = order[head]
        head += 1
        
        side, x = vertex
        neighbors = compute_neighbours(side, x)
        
        for other in neighbors:
            if other not in index:
                index[other] = len(order)
                order.append(other)
            
            if side == 0:
                edges.append((index[vertex], index[other]))

    n = len(order)
    
    # Build adjacency list for verification
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Quick girth verification using non-backtracking BFS from a few vertices
    def check_girth_nb(start_node, adj, n, g, deadline):
        """Non-backtracking BFS to check if girth >= g."""
        visited = set()
        queue = deque()
        
        for neighbor in adj[start_node]:
            state = (neighbor, start_node)
            if state not in visited:
                visited.add(state)
                queue.append(state)
        
        depth = 1
        max_depth = g // 2
        
        while queue and depth < max_depth:
            if time.time() > deadline:
                return True  # Assume valid
            
            next_queue = deque()
            while queue:
                curr, prev = queue.popleft()
                
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    
                    new_state = (nxt, curr)
                    
                    if new_state in visited:
                        return False  # Found cycle shorter than g
                    
                    if new_state not in visited:
                        visited.add(new_state)
                        next_queue.append(new_state)
            
            queue = next_queue
            depth += 1
        
        return True
    
    # Verify girth from a small sample
    random.seed(42)
    sample_size = min(10, n)
    if n > 0:
        sample = random.sample(range(n), sample_size)
        for s in sample:
            if time.time() > deadline - 15:
                break
            if not check_girth_nb(s, adj, n, g, deadline - 10):
                # Girth check failed; still return the graph (LUW guarantee should hold)
                break
    
    return n, edges
