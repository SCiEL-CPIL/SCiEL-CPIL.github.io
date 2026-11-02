import time
import random
import sys
from collections import deque

_DEADLINE = 145.0

def _generate_random_regular(n, k):
    if n * k % 2 != 0:
        return None
    half_edges = []
    for i in range(n):
        for _ in range(k):
            half_edges.append(i)
    
    max_retries = 1000
    for _ in range(max_retries):
        random.shuffle(half_edges)
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if valid:
            return adj
    return None

def _find_short_cycle(adj, g):
    n = len(adj)
    # We want to find a cycle of length < g
    # Max depth needed: floor((g-1)/2)
    max_depth = (g - 1) // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v == parent[u]:
                        continue
                    # Check if cycle length < g
                    L = dist[u] + dist[v] + 1
                    if L < g:
                        return (u, v, L)
    return None

def _try_switching(k, g, n, time_limit):
    start_time = time.time()
    adj = _generate_random_regular(n, k)
    if adj is None:
        return None
    
    # Check initial girth
    while time.time() - start_time < time_limit:
        cycle = _find_short_cycle(adj, g)
        if cycle is None:
            # Success
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
        
        u, v, L = cycle
        # Try to switch edge (u, v) with a random edge (x, y)
        success = False
        # To improve success rate, try multiple random switches
        for _ in range(50):
            if time.time() - start_time >= time_limit:
                break
            x = random.randint(0, n - 1)
            if x == u or x == v:
                continue
            if len(adj[x]) == 0:
                continue
            y = random.choice(list(adj[x]))
            if y == u or y == v or y == x:
                continue
            if y in adj[v] or x in adj[u]:
                continue
            
            # Perform switch
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            
            success = True
            break
            
        if not success:
            # If we can't find a valid switch, maybe restart?
            # Or just continue, the next iteration might find a different cycle
            # But if we can't switch, we are stuck.
            # Let's try a different edge in the cycle?
            # For now, just restart the whole process if we fail too many times?
            # No, just continue.
            pass

    return None

def construct(k, g):
    global _DEADLINE
    if k == 5 and g == 9:
        # Try sizes from small to large?
        # Or large to small?
        # If seed is 15625, even 1500 is a win.
        # Let's try 1500 first to ensure we get a valid graph.
        # Then try smaller.
        
        remaining = _DEADLINE - (time.time() - 0) # Roughly
        # Allocate time for attempts
        
        # Try 1116 (best known)
        # If it fails, try 1200, 1300, 1500
        
        target_ns = [1116, 1200, 1300, 1500]
        
        # Give more time to smaller n?
        # Smaller n is harder to satisfy girth.
        
        for n in target_ns:
            if time.time() > _DEADLINE - 10:
                break
            # Allocate time based on n?
            # Just use a fixed chunk
            t_limit = 20.0
            res = _try_switching(k, g, n, t_limit)
            if res is not None:
                return res
        
        # Fallback to LUW if all fail
        # But LUW is 15625.
        # If we haven't found anything, return LUW?
        # The current program returns LUW.
        # I will implement a basic LUW or just return a large random graph?
        # No, must be valid.
        # If switching fails, we can try a very large n where girth is almost certainly >= 9.
        # e.g. n=5000.
        # Random 5-reg on 5000 has girth ~ 2 log4(5000) ~ 11.
        # So it's very likely valid.
        
        res = _try_switching(k, g, 5000, 10.0)
        if res is not None:
            return res
            
        # Last resort: LUW
        return _luw_construct(k, g)
        
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # Same as before
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
