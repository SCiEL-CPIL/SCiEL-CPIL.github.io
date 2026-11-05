"""Cayley-graph construction for 4-regular girth-10 graph."""
import itertools
import numpy as np
import time
import math

def construct(k, g):
    # We construct a 4-regular graph on a finite abelian group Z_m^d (or similar)
    # using a symmetric 4-element generating set, and verify girth >= g.
    # For k=4, g=10, we search over small abelian groups.
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    if k != 4:
        # Fallback: use a simple construction for other k
        return _construct_fallback(k, g)
    
    best_n = None
    best_edges = None
    
    # Try cyclic groups Z_m for m from Moore bound upward
    # The Moore bound for (4,10) is 243, so n >= 243.
    # Try m from 243 to ~400, but with efficient checks
    
    for m in range(243, 420):
        if time.time() > deadline - 5.0:
            break
        
        # Try generating sets {a, b, -a, -b} with 1 <= a < b < m/2
        found = False
        for a in range(1, m // 2 + 1):
            if time.time() > deadline - 5.0:
                break
            if a == m - a:
                continue  # would create 2-cycles
            for b in range(a + 1, m // 2 + 1):
                if time.time() > deadline - 5.0:
                    break
                if b == m - b:
                    continue
                if a == b or a == m - b or b == m - a:
                    continue
                # Check connectedness: gcd(a, b, m) == 1
                if math.gcd(math.gcd(a, b), m) != 1:
                    continue
                
                # Efficient girth check
                if _check_girth_fast(m, a, b, g):
                    # Construct edges
                    edge_set = set()
                    for i in range(m):
                        for s in [a, b]:
                            j = (i + s) % m
                            if i < j:
                                edge_set.add((i, j))
                            else:
                                edge_set.add((j, i))
                    edges = sorted(edge_set)
                    
                    if best_n is None or m < best_n:
                        best_n = m
                        best_edges = edges
                    found = True
                    break
            if found:
                break
        if found:
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # If no cyclic group works, try Z_a x Z_b
    # This is more complex; for now, return a fallback
    return _construct_fallback(4, g)


def _check_girth_fast(m, a, b, g):
    """Efficient girth check for Cayley graph of Z_m with S={a,-a,b,-b}.
    Checks that there are no non-backtracking closed walks of length < g.
    """
    S = [a, -a, b, -b]
    
    for L in range(2, g):
        if _has_nonbacktracking_closed_walk(m, S, L):
            return False
    
    return True


def _has_nonbacktracking_closed_walk(m, S, L):
    """Check if there's a non-backtracking closed walk of length L."""
    def _rec(depth, current_sum, prev):
        if depth == L:
            return current_sum % m == 0
        for s in S:
            if s == -prev:
                continue
            if _rec(depth + 1, current_sum + s, s):
                return True
        return False
    
    for s in S:
        if _rec(1, s, s):
            return True
    return False


def _construct_fallback(k, g):
    """Fallback: use the Lazebnik-Ustimenko-Woldar construction from the seed."""
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
                other_key = other
                index[other_key] = len(order)
                order.append(other_key)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
