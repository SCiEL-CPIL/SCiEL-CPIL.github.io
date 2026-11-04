# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

_DEADLINE = 120.0
_MARGIN = 5.0

def _validate_graph(n, edges, k, g, max_vertices=None):
    """
    Validate simple k-regular graph with girth >= g.
    Returns True if valid, False otherwise.
    """
    if max_vertices is not None and n > max_vertices:
        return False
        
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # Check girth using non-backtracking BFS from each vertex
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
            
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
                
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
                    
    return True

def _luw_graph(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Returns (n, edges).
    """
    q = k
    d = g - 5
    if d < 3:
        d = 3
        
    if q == 4:
        table = [[0, 0, 0, 0], 
                 [0, 1, 2, 3], 
                 [0, 2, 3, 1], 
                 [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

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

def _find_circulant_graph(k, g, max_n_search=1000):
    """
    Search for a circulant k-regular graph with girth >= g.
    Optimized for k=4.
    """
    if k % 2 != 0:
        return None, None
        
    num_gens = k // 2
    
    # Calculate Moore bound
    if g % 2 == 0:
        m = g // 2 - 1
    else:
        m = (g - 1) // 2 - 1
        
    if m < 0:
        m = 0
        
    moore_bound = 1 + k * sum((k-1)**i for i in range(m+1))
    
    start_n = max(k + 1, moore_bound)
    
    if num_gens != 2:
        # Fallback for other k: just return None and let LUW handle it
        # Or implement general case later
        return None, None

    for n in range(start_n, max_n_search + 1):
        for a in range(1, n // 2 + 1):
            for b in range(a + 1, n // 2 + 1):
                # Check simplicity conditions
                if 2 * a % n == 0 or 2 * b % n == 0:
                    continue
                if a == b or a == n - b:
                    continue
                
                # Check girth via non-backtracking BFS from 0
                visited = set()
                dq = deque()
                gens = [a, n - a, b, n - b]
                for s in gens:
                    if s % n == 0:
                        continue
                    state = (s, 0)
                    if state not in visited:
                        visited.add(state)
                        dq.append((s, 0, 1))
                        
                found_short_cycle = False
                while dq:
                    curr, prev, depth = dq.popleft()
                    # We want cycle length < g.
                    # Cycle length = depth + 1.
                    # So we care if depth + 1 < g => depth < g - 1.
                    if depth >= g - 1:
                        continue
                        
                    for s in gens:
                        next_v = (curr + s) % n
                        if next_v == prev:
                            continue
                            
                        if next_v == 0:
                            if depth + 1 < g:
                                found_short_cycle = True
                                break
                            
                        state = (next_v, curr)
                        if state not in visited:
                            visited.add(state)
                            dq.append((next_v, curr, depth + 1))
                            
                    if found_short_cycle:
                        break
                        
                if not found_short_cycle:
                    edges = []
                    for i in range(n):
                        for s in [a, b]:
                            j = (i + s) % n
                            edges.append((i, j))
                    return n, edges
                        
    return None, None

def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    # 1. Try Circulant Graph Search (fast, deterministic)
    n_circ, edges_circ = _find_circulant_graph(k, g)
    
    if n_circ is not None:
        # Verify just in case
        if _validate_graph(n_circ, edges_circ, k, g):
            return n_circ, edges_circ

    # 2. Try LUW Graph (fallback, guaranteed valid but large)
    n_luw, edges_luw = _luw_graph(k, g)
    
    return n_luw, edges_luw

# EVOLVE-BLOCK-END
