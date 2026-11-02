"""Construct a 4-regular graph of girth 10 (the (4,10)-cage problem)."""
import numpy as np
from collections import deque
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph of girth at least g.
    
    For k=4, g=10, returns a 4-regular graph with girth at least 10.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction.
    """
    if k == 4 and g == 10:
        # Use LUW with q=4, d=4.
        # Size: 2 * 4^4 = 512.
        # Girth: 2*d + 2 = 10.
        return _luw_construct(4, 4)
    else:
        return _construct_general(k, g)

def _construct_general(k, g):
    """General construction for other parameters using LUW method."""
    if g % 2 == 0:
        d = g // 2 - 1
        # Find smallest prime power q >= k
        q = k
        while not _is_prime_power(q):
            q += 1
        return _luw_construct(q, d)
    else:
        # Fallback for odd girth
        n = max(2 * k, g)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if j - i <= k // 2 or n - (j - i) <= k // 2:
                    edges.append((i, j))
        return n, edges

def _is_prime_power(n):
    """Check if n is a prime power."""
    if n < 2:
        return False
    for p in range(2, int(n ** 0.5) + 1):
        if n % p == 0:
            m = n
            while m % p == 0:
                m //= p
            return m == 1
    return True

def _luw_construct(q, d):
    """LUW construction for q-regular bipartite graph of girth 2d+2."""
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    if q == 4:
        table = [[0, 0, 0, 0],
                 [0, 1, 2, 3],
                 [0, 2, 3, 1],
                 [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q == 2:
        table = [[0, 0], [0, 1]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: x & y
    else:
        table = [[0]*q for _ in range(q)]
        for i in range(q):
            for j in range(q):
                table[i][j] = (i * j) % q
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: table[x][y]
    
    # Rule for LUW: y[c] = f(y[a], x[b])
    # Standard rule for girth 2d+2
    rule = {}
    for c in range(2, d + 1):
        if c % 2 == 0:
            rule[c] = (1, c - 1)
        else:
            rule[c] = (c - 1, 1)
    
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
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if time.time() > deadline:
                break
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
