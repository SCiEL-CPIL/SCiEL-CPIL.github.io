import time
import random
import sys
from collections import deque

_DEADLINE = 145.0
_start_time_global = time.time()

def _elapsed():
    return time.time() - _start_time_global

def _remaining_time():
    return _DEADLINE - _elapsed()

def _generate_random_regular(n, k):
    if n * k % 2 != 0:
        return None
    half_edges = []
    for i in range(n):
        for _ in range(k):
            half_edges.append(i)
    
    max_retries = 50
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
    """Find a cycle of length < g using BFS from each vertex."""
    n = len(adj)
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
                    L = dist[u] + dist[v] + 1
                    if L < g:
                        return (u, v, L)
    return None

def _try_switching(k, g, n, time_limit):
    """Try to construct a k-regular graph with girth >= g on n vertices via edge switching."""
    start_time = time.time()
    adj = _generate_random_regular(n, k)
    if adj is None:
        return None
    
    # Build edge list for faster switching
    edge_set = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_set.add((u, v))
    
    # Track how many times we've failed to switch
    fail_count = 0
    
    while time.time() - start_time < time_limit:
        cycle = _find_short_cycle(adj, g)
        if cycle is None:
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
        
        u, v, L = cycle
        # Try multiple random 2-switches to break this cycle
        max_attempts = 100
        success = False
        for _ in range(max_attempts):
            if time.time() - start_time >= time_limit:
                break
            
            # Pick a random edge (x, y) that doesn't share vertices with (u, v)
            # and doesn't create multi-edges
            x = random.randint(0, n - 1)
            if x == u or x == v:
                continue
            
            # Pick a random neighbor y of x
            if not adj[x]:
                continue
            y = random.choice(list(adj[x]))
            
            if y == u or y == v or y == x:
                continue
            if (x, u) in edge_set or (u, x) in edge_set:
                continue
            if (y, v) in edge_set or (v, y) in edge_set:
                continue
            
            # Perform the 2-switch
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            
            edge_set.discard((u, v))
            edge_set.discard((v, u))
            edge_set.discard((x, y))
            edge_set.discard((y, x))
            edge_set.add((u, x) if u < x else (x, u))
            edge_set.add((v, y) if v < y else (y, v))
            
            success = True
            break
        
        if not success:
            fail_count += 1
            if fail_count > 50:
                # Restart with a fresh random graph
                adj = _generate_random_regular(n, k)
                if adj is None:
                    return None
                edge_set = set()
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edge_set.add((u, v))
                fail_count = 0
    
    return None

def _luw_construct(k, g):
    """Construct a large girth graph using the LUW method."""
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

def construct(k, g):
    if k == 5 and g == 9:
        # Try progressively larger graphs, allocating time based on remaining budget
        # Start with smaller targets and increase if they fail
        
        # Moore bound for (5,9): 2*(4^4 - 1)/3 = 2*255/3 = 170, but actual lower bound is 428
        # Best known is 1116, so try values around there
        
        target_ns = [1000, 1100, 1200, 1400, 1800, 2500]
        
        for n in target_ns:
            if _remaining_time() < 15:
                break
            
            # Allocate more time for smaller n (harder to achieve girth)
            if n <= 1100:
                t_limit = min(25.0, _remaining_time() - 10)
            elif n <= 1500:
                t_limit = min(20.0, _remaining_time() - 10)
            elif n <= 2000:
                t_limit = min(15.0, _remaining_time() - 10)
            else:
                t_limit = min(10.0, _remaining_time() - 10)
            
            if t_limit <= 0:
                break
            
            res = _try_switching(k, g, n, t_limit)
            if res is not None:
                return res
        
        # Fallback: try a large random graph where girth is almost certainly >= 9
        if _remaining_time() > 10:
            res = _try_switching(k, g, 5000, 8.0)
            if res is not None:
                return res
        
        # Last resort: LUW construction
        return _luw_construct(k, g)
    
    return _luw_construct(k, g)
