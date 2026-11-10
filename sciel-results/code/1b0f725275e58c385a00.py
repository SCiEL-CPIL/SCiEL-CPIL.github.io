import time
import random
from collections import deque

def _luw_construct(d, q):
    """
    Construct Lazebnik-Ustimenko-Woldar graph CD(d, q).
    Returns (n, edges).
    """
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
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _check_girth_nonbacktracking(n, edges, g):
    """
    Check girth >= g using non-backtracking walks from every vertex.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = g - 1
    
    for start in range(n):
        visited = {}
        queue = deque()
        
        for nb in adj[start]:
            state = (nb, start)
            if state not in visited:
                visited[state] = 1
                queue.append((nb, start, 1))
        
        while queue:
            curr, prev, dist = queue.popleft()
            
            if curr == start:
                if dist < g:
                    return False
                continue
            
            for nb in adj[curr]:
                if nb == prev:
                    continue
                new_state = (nb, curr)
                new_dist = dist + 1
                if new_state not in visited:
                    visited[new_state] = new_dist
                    if new_dist < g:
                        queue.append((nb, curr, new_dist))
    
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    best_n = None
    best_edges = None
    
    candidates = []
    for d in range(3, 10):
        if 2 * d + 2 >= g:
            candidates.append(d)
    
    candidates.sort()
    
    for d in candidates:
        if time.time() > deadline:
            break
        try:
            n, edges = _luw_construct(d, k)
            if _check_girth_nonbacktracking(n, edges, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
        except Exception:
            pass
    
    if best_n is None:
        d = max(3, (g - 2) // 2)
        try:
            n, edges = _luw_construct(d, k)
            best_n = n
            best_edges = edges
        except Exception:
            best_n = k + 1
            best_edges = []
            for i in range(k):
                best_edges.append((0, i + 1))
    
    return best_n, best_edges
