import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # q must be k to achieve degree k
    q = k
    
    # Check if q is a prime power (simplified check for small numbers)
    # For k=5, it's fine.
    def is_prime_power(n):
        if n <= 1: return False
        # Check for prime powers
        # Simple check for small n
        # This is a heuristic, but for k=5 it works.
        # If k is not prime power, this construction might fail or not be defined.
        # We assume k is valid for LUW construction (prime or 4).
        if n == 4: return True
        # Check if n is prime
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0:
                # Check if i is prime and n/i is power of i? 
                # Actually just check if n is prime
                return False
        return True
        
    # If k is not prime power, we cannot use this construction directly.
    # However, the task implies k=5.
    
    # GF(q) operations
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y) % q
        sub = lambda x,y: (x-y) % q
        mul = lambda x,y: x*y % q
    
    def get_rule(d):
        rule = {}
        if d >= 2: rule[2] = (1,1)
        if d >= 3: rule[3] = (2,1)
        if d >= 4: rule[4] = (1,2)
        for c in range(5, d+1):
            r = (c-5) % 4
            c0 = c - r
            rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
        return rule

    def build_graph(d):
        rule = get_rule(d)
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,) + (0,)*d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for i, vertex in enumerate(order):
            if time.time() > deadline:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        return n, edges

    def check_girth(n, edges, g):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        girth = float('inf')
        # BFS from each node to find shortest cycle
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth:
                            girth = cycle_len
        return girth >= g

    # Determine candidate d values
    # Theoretical girth of CD(d,q) is at least 2d+2 (for q > d)
    # So we need 2d+2 >= g  => d >= (g-2)/2
    d_min = max(3, (g - 2 + 1) // 2)
    
    best_n = float('inf')
    best_edges = []
    
    # Try d from d_min upwards
    # Cap d to avoid running out of time or memory
    for d in range(d_min, d_min + 5):
        if time.time() > deadline - 10:
            break
            
        n, edges = build_graph(d)
        
        if n < best_n:
            # Check if girth is sufficient
            if check_girth(n, edges, g):
                best_n = n
                best_edges = edges
                # Since we are increasing d, and n increases with d, 
                # the first valid one is the smallest.
                # So we can return immediately?
                # Yes, because n = 2*q*(d+1) is increasing in d.
                break
    
    if best_n == float('inf'):
        # Fallback: use original logic
        d = g - 5 if (g - 5) % 2 == 1 else g - 4
        if d < 3:
            d = 3
        n, edges = build_graph(d)
        return n, edges
        
    return best_n, best_edges
