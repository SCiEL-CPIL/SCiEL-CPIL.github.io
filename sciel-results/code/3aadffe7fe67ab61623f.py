import time
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g.
    For k=3, uses LUW construction with d chosen to satisfy girth.
    """
    if k != 3:
        return _luw_construct(k, g)
    return _luw_construct(3, g)

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction.
    """
    if k == 3:
        q = 3
        # Need 2d - 1 >= g
        d = (g + 1) // 2
        if 2 * d - 1 < g:
            d += 1
        return _build_luw(q, d)
    else:
        # Fallback for other k (not used in this task)
        raise NotImplementedError("Only k=3 supported")

def _build_luw(q, d):
    """
    Builds the LUW graph for parameters q, d.
    """
    add = lambda x, y: (x + y) % q
    sub = lambda x, y: (x - y) % q
    mul = lambda x, y: (x * y) % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    index = {}
    order = []
    edges = []
    
    start = (0, (0,) * d)
    index[start] = 0
    order.append(start)
    
    for vertex in order:
        side, x = vertex
        for y1 in range(q):
            y = [0] * (d + 1)
            y[1] = y1
            for c in range(2, d + 1):
                a, b = rule[c]
                val = mul(x[a-1], x[b-1])
                if side == 0:
                    y[c] = add(x[c-1], val)
                else:
                    y[c] = sub(x[c-1], val)
            
            neighbor = (1 - side, tuple(y[1:]))
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            
            if side == 0:
                edges.append((index[vertex], index[neighbor]))
                
    return len(order), edges
