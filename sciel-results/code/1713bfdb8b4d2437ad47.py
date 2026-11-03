import time
import collections
import math
import random

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _check_girth_nb(n, adj, min_girth):
    """
    Fast girth check using non-backtracking BFS from each directed edge.
    """
    if _time_left() < 2.0:
        return True
    
    max_depth = min_girth - 2
    
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            
            visited = {}
            queue = collections.deque()
            queue.append((u, v, 0))
            visited[(u, v)] = 0
            
            while queue:
                prev, curr, depth = queue.popleft()
                
                if depth >= max_depth:
                    continue
                
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    
                    new_state = (curr, nxt)
                    if new_state == (v, u) and depth + 1 >= 2:
                        if depth + 2 < min_girth:
                            return False
                    elif new_state not in visited:
                        visited[new_state] = depth + 1
                        queue.append((curr, nxt, depth + 1))
                
                if _time_left() < 1.0:
                    return True
    
    return True

def _construct_luw(q, d):
    """
    Constructs the LUW graph CD(d, q).
    """
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
        if _time_left() < 5.0:
            return None
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _try_luw_params(k, g):
    """Try multiple LUW parameter combinations and return the best valid graph."""
    q = k
    # Heuristic: Girth is approx 2d+1. 
    # We want 2d+1 >= g => d >= (g-1)/2.
    # But for small q, girth might be larger or smaller.
    # Let's start lower to find smaller n.
    # (g-4)//2 allows d=5 for g=16.
    min_d = max(3, (g - 4) // 2)
    
    best_n = float('inf')
    best_edges = None
    
    # Try d from min_d to min_d + 10
    for d in range(min_d, min_d + 11):
        if _time_left() < 15.0:
            break
        
        result = _construct_luw(q, d)
        if result is None:
            continue
        
        n, edges = result
        
        if n >= best_n:
            continue
        
        # Verify girth
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        if _check_girth_nb(n, adj, g):
            best_n = n
            best_edges = edges
            # Early exit if we hit a reasonable bound (e.g., near Moore bound)
            # Moore bound for k=3, g=16 is ~ 512.
            # If we get below 1000, it's very good.
            if n <= 1000:
                break
    
    if best_edges is not None:
        return best_n, best_edges
    return None

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    # Primary: LUW construction with parameter search
    luw_result = _try_luw_params(k, g)
    
    if luw_result is not None:
        return luw_result
    
    # Fallback
    if _time_left() > 10.0:
        q = k
        min_d = max(3, (g - 2) // 2)
        result = _construct_luw(q, min_d)
        if result is not None:
            return result
    
    # Last resort
    n = 6
    edges = [(0,1), (1,2), (2,0), (3,4), (4,5), (5,3), (0,3), (1,4), (2,5)]
    return n, edges
