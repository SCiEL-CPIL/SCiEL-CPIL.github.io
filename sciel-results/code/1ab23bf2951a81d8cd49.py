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
    
    max_retries = 500
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

def _check_girth_nonbacktracking(adj, g):
    """Check if graph has girth >= g using non-backtracking walks from each vertex.
    Returns (True, None) if girth >= g, else (False, (u, v, L)) for a short cycle."""
    n = len(adj)
    max_depth = (g - 1) // 2
    
    for start in range(n):
        # dist[v] = shortest non-backtracking path length from start to v
        # prev[v] = predecessor on that path (for non-backtracking)
        dist = [None] * n
        prev = [None] * n
        dist[start] = 0
        queue = deque([(start, -1)])  # (vertex, predecessor)
        
        while queue:
            u, pred = queue.popleft()
            d = dist[u]
            if d >= max_depth:
                continue
            for v in adj[u]:
                if v == pred:
                    continue
                if dist[v] is None:
                    dist[v] = d + 1
                    prev[v] = u
                    queue.append((v, u))
                else:
                    # Found a cycle
                    # The non-backtracking path from start to v has length dist[v]
                    # The edge (u, v) creates a cycle of length d + dist[v] + 1
                    # But we need to be careful: this might not be the shortest cycle through v
                    L = d + dist[v] + 1
                    if L < g:
                        return (False, (u, v, L))
    return (True, None)

def _find_all_short_cycles(adj, g, max_cycles=10):
    """Find up to max_cycles short cycles (< g) using non-backtracking walks."""
    n = len(adj)
    max_depth = (g - 1) // 2
    cycles = []
    
    for start in range(n):
        if len(cycles) >= max_cycles:
            break
        dist = [None] * n
        prev = [None] * n
        dist[start] = 0
        queue = deque([(start, -1)])
        
        while queue and len(cycles) < max_cycles:
            u, pred = queue.popleft()
            d = dist[u]
            if d >= max_depth:
                continue
            for v in adj[u]:
                if v == pred:
                    continue
                if dist[v] is None:
                    dist[v] = d + 1
                    prev[v] = u
                    queue.append((v, u))
                else:
                    L = d + dist[v] + 1
                    if L < g:
                        cycles.append((u, v, L))
                        if len(cycles) >= max_cycles:
                            break
    return cycles

def _try_switching(k, g, n, time_limit, start_time=None):
    if start_time is None:
        start_time = time.time()
    
    adj = _generate_random_regular(n, k)
    if adj is None:
        return None
    
    # Check initial girth
    stuck_count = 0
    while time.time() - start_time < time_limit:
        ok, cycle = _check_girth_nonbacktracking(adj, g)
        if ok:
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
        
        u, v, L = cycle
        # Try to find a good 2-switch that eliminates this cycle
        success = False
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() - start_time >= time_limit:
                break
            x = random.randint(0, n - 1)
            if x == u or x == v:
                continue
            if not adj[x]:
                continue
            y = random.choice(list(adj[x]))
            if y == u or y == v or y == x:
                continue
            if y in adj[v] or x in adj[u]:
                continue
            
            # Perform switch
            adj[u].discard(v)
            adj[v].discard(u)
            adj[x].discard(y)
            adj[y].discard(x)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            
            success = True
            break
        
        if not success:
            stuck_count += 1
            if stuck_count > 50:
                # Restart with a new random graph
                adj = _generate_random_regular(n, k)
                if adj is None:
                    return None
                stuck_count = 0
    
    return None

def construct(k, g):
    global _DEADLINE
    t0 = time.time()
    
    if k == 5 and g == 9:
        # Try sizes from small to large
        # Smaller n is harder but better score
        target_ns = [1116, 1200, 1400, 1600]
        
        # Allocate time: give more time to smaller n
        # Total budget: _DEADLINE - t0 - 5 (safety margin)
        total_budget = _DEADLINE - t0 - 5.0
        
        # Try each n with decreasing time allocation
        # First attempt gets most time, subsequent ones less
        time_allocs = [0.35, 0.25, 0.20, 0.15]
        
        for i, n in enumerate(target_ns):
            if time.time() - t0 > _DEADLINE - 10:
                break
            t_limit = time_allocs[i] * total_budget
            if t_limit < 5.0:
                t_limit = 5.0
            res = _try_switching(k, g, n, t_limit, start_time=t0)
            if res is not None:
                return res
        
        # Fallback: try a larger n where random graph likely has girth >= 9
        # Random 5-reg on n vertices has girth ~ 2*log_4(n)
        # For girth >= 9: n >= 4^(4.5) ≈ 1024
        # So n=2000 should be quite safe
        if time.time() - t0 < _DEADLINE - 15:
            res = _try_switching(k, g, 2000, 15.0, start_time=t0)
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
