import time
import random
import heapq
from collections import deque
import numpy as np

def _luw_construct(k, g):
    # Fallback to the original LUW construction
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

def _compute_dist_matrix(adjacency, n, max_dist):
    """Compute shortest path distances up to max_dist from every vertex."""
    # dist[i][j] = shortest distance from i to j, or INF if > max_dist
    INF = max_dist + 1
    dist = np.full((n, n), INF, dtype=np.int8)
    for i in range(n):
        dist[i][i] = 0
        queue = deque([i])
        dist[i][i] = 0
        while queue:
            u = queue.popleft()
            d_u = dist[i][u]
            if d_u >= max_dist:
                continue
            for v in adjacency[u]:
                if dist[i][v] == INF:
                    dist[i][v] = d_u + 1
                    queue.append(v)
    return dist

def _generate_simple_k_regular(k, n, max_attempts=100):
    """Generate a simple k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None
    
    for attempt in range(max_attempts):
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        
        random.shuffle(stubs)
        
        adjacency = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v or v in adjacency[u]:
                valid = False
                break
            adjacency[u].add(v)
            adjacency[v].add(u)
        
        if valid:
            for i in range(n):
                if len(adjacency[i]) != k:
                    valid = False
                    break
            if valid:
                return [sorted(adjacency[i]) for i in range(n)]
    
    return None

def _switching_construct_fast(k, g, n, time_limit=100.0):
    """Construct a k-regular graph of girth >= g using fast edge switching."""
    start_time = time.time()
    
    adjacency = _generate_simple_k_regular(k, n, max_attempts=100)
    if adjacency is None:
        return None
    
    adj_sets = [set(adj) for adj in adjacency]
    
    # Precompute distances up to g-2
    max_d = g - 2
    dist = _compute_dist_matrix(adjacency, n, max_d)
    
    # Helper to check if a switch is valid using dist matrix
    # Switch (u, v) -> (u, w) and (v, x)
    # We need to ensure no short cycles are created.
    # A new cycle involving (u, w) requires a path from u to w of length <= g-2 in G \ {(u,v), (w,x)}
    # The dist matrix is for G.
    # If dist[u][w] > g-2, it's likely safe.
    # If dist[u][w] <= g-2, it might be unsafe, but we need to check if the shortest path uses the removed edges.
    
    def is_switch_valid(u, v, w, x):
        # Check if path u->w or v->x is short in G \ {(u,v), (w,x)}
        # We do a local BFS for verification, but use dist to filter
        if dist[u][w] <= max_d or dist[v][x] <= max_d:
             # Potential conflict, verify with BFS
             # BFS from u to w avoiding (u,v) and (w,x)
             # And BFS from v to x avoiding (u,v) and (w,x)
             # Also need to check if other short cycles are created, but this is the main risk
             pass
        else:
             return True
             
        # Verification
        # BFS u to w
        visited = {u: 0}
        queue = deque([(u, 0)])
        while queue:
            node, d = queue.popleft()
            if d >= max_d: continue
            for neighbor in adjacency[node]:
                if (node == u and neighbor == v) or (node == w and neighbor == x) or (node == x and neighbor == w) or (node == v and neighbor == u):
                    continue
                if neighbor == w:
                    return False
                if neighbor not in visited:
                    visited[neighbor] = d + 1
                    queue.append((neighbor, d + 1))
                    
        # BFS v to x
        visited = {v: 0}
        queue = deque([(v, 0)])
        while queue:
            node, d = queue.popleft()
            if d >= max_d: continue
            for neighbor in adjacency[node]:
                if (node == u and neighbor == v) or (node == w and neighbor == x) or (node == x and neighbor == w) or (node == v and neighbor == u):
                    continue
                if neighbor == x:
                    return False
                if neighbor not in visited:
                    visited[neighbor] = d + 1
                    queue.append((neighbor, d + 1))
                    
        return True

    # Main loop
    while time.time() - start_time < time_limit:
        # Find a short cycle
        # Scan edges for dist[u][v] <= max_d
        short_edge = None
        for u in range(n):
            for v in adjacency[u]:
                if u < v and dist[u][v] <= max_d:
                    short_edge = (u, v)
                    break
            if short_edge:
                break
                
        if short_edge is None:
            # Verify girth
            # If no edge has a short alternative path, girth is >= g
            return n, [(u, v) for u in range(n) for v in adjacency[u] if u < v]
            
        u, v = short_edge
        
        # Try to switch (u, v) with (w, x)
        success = False
        for _ in range(50): # Try 50 random switches
            w = random.randint(0, n - 1)
            if not adjacency[w]: continue
            x = random.choice(adjacency[w])
            
            # Basic checks
            if w == u or w == v or x == u or x == v:
                continue
            if x in adj_sets[u] or w in adj_sets[v]:
                continue
            if u in adj_sets[w] or v in adj_sets[x]:
                continue
            
            # Quick filter using dist
            if dist[u][w] <= max_d or dist[v][x] <= max_d:
                # Might be invalid, but check
                if is_switch_valid(u, v, w, x):
                    # Perform switch
                    adj_sets[u].remove(v)
                    adj_sets[v].remove(u)
                    adj_sets[w].remove(x)
                    adj_sets[x].remove(w)
                    
                    adj_sets[u].add(w)
                    adj_sets[w].add(u)
                    adj_sets[v].add(x)
                    adj_sets[x].add(v)
                    
                    adjacency[u] = sorted(adj_sets[u])
                    adjacency[v] = sorted(adj_sets[v])
                    adjacency[w] = sorted(adj_sets[w])
                    adjacency[x] = sorted(adj_sets[x])
                    
                    # Recompute dist
                    dist = _compute_dist_matrix(adjacency, n, max_d)
                    success = True
                    break
        
        if not success:
            # If no switch worked, try a random valid switch to escape local minima
            # Or just break if stuck
            if random.random() < 0.1:
                # Try a random switch that is definitely valid (dist > max_d)
                for _ in range(10):
                    w = random.randint(0, n - 1)
                    if not adjacency[w]: continue
                    x = random.choice(adjacency[w])
                    if w in (u, v) or x in (u, v): continue
                    if x in adj_sets[u] or w in adj_sets[v]: continue
                    if u in adj_sets[w] or v in adj_sets[x]: continue
                    if dist[u][w] > max_d and dist[v][x] > max_d:
                         # Likely valid
                         if is_switch_valid(u, v, w, x):
                             adj_sets[u].remove(v)
                             adj_sets[v].remove(u)
                             adj_sets[w].remove(x)
                             adj_sets[x].remove(w)
                             
                             adj_sets[u].add(w)
                             adj_sets[w].add(u)
                             adj_sets[v].add(x)
                             adj_sets[x].add(v)
                             
                             adjacency[u] = sorted(adj_sets[u])
                             adjacency[v] = sorted(adj_sets[v])
                             adjacency[w] = sorted(adj_sets[w])
                             adjacency[x] = sorted(adj_sets[x])
                             
                             dist = _compute_dist_matrix(adjacency, n, max_d)
                             break
                             
    return None

def construct(k, g):
    if k == 5 and g == 9:
        total_time = 140.0
        start_time = time.time()
        
        # Try smaller n first
        target_ns = [1116, 1200, 1500]
        
        for n in target_ns:
            elapsed = time.time() - start_time
            remaining = total_time - elapsed
            if remaining < 20:
                break
            time_per_attempt = min(40.0, remaining / 2)
            
            res = _switching_construct_fast(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
        
    return _luw_construct(k, g)
