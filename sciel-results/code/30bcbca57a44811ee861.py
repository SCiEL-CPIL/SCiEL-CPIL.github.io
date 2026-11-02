import math
import time
import numpy as np

def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)
    
    r = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(r))
    if moore_bound % 2 != 0:
        moore_bound += 1
    
    start_time = time.time()
    deadline = 105  # seconds, safety margin
    
    # Search for circulant C(n; {a, n-a, n/2}) with girth >= g
    max_n = 1500
    for n in range(moore_bound, max_n + 1, 2):
        if time.time() - start_time > deadline:
            break
        
        h = n // 2
        # gcd(n, a, h) = gcd(h, a) must be 1 for connectivity
        for a in range(1, h):
            if math.gcd(h, a) != 1:
                continue
            
            if _check_girth_circulant(n, a, g, start_time, deadline):
                edges = []
                for u in range(n):
                    for s in (a, n - a, h):
                        v = (u + s) % n
                        if u < v:
                            edges.append((u, v))
                return n, edges
    
    return _luw_fallback(k, g)

def _check_girth_circulant(n, a, g, start_time, deadline):
    """Check if C(n; {a, -a, n/2}) has girth >= g using non-backtracking walks."""
    h = n // 2
    S = [a, n - a, h]
    neg_idx = [1, 0, 2]
    
    R = np.zeros((3, n), dtype=bool)
    R[0, a % n] = True
    R[1, (n - a) % n] = True
    R[2, h % n] = True
    
    for d in range(2, g):
        if time.time() - start_time > deadline:
            return False
        R_new = np.zeros((3, n), dtype=bool)
        for i in range(3):
            for j in range(3):
                if j == neg_idx[i]:
                    continue
                R_new[j] |= np.roll(R[i], S[j])
        R = R_new
        if R[:, 0].any():
            return False
    
    return True

def _luw_fallback(k, g):
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
