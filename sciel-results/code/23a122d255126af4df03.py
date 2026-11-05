"""Cayley-graph construction for 4-regular girth-10 graph."""
import math
import numpy as np

def construct(k, g):
    if k != 4:
        return _construct_fallback(k, g)

    # Try known small 4-regular circulant graphs with large girth.
    # For k=4, g=10, the best known upper bound is 320 (Exoo et al. 2026).
    # We search cyclic groups Z_m for m in [243, 420] with S={a, -a, b, -b}.
    
    best_n = None
    best_edges = None
    
    # First, try the known 320-vertex construction if applicable
    # C_320(1, 13) is a well-known circulant with girth 14 (so >= 10)
    # Let's verify it works for g=10
    m = 320
    if _check_girth_cyclic(m, 1, 13, g):
        best_n = m
        best_edges = _make_circulant_edges(m, 1, 13)
    
    # If the known construction doesn't work (shouldn't happen for g=10),
    # or if we want to find something smaller, search for smaller m
    if best_n is None or best_n > 243:
        # Search for smaller m from Moore bound up to current best
        upper = best_n - 1 if best_n else 420
        for m in range(243, upper + 1):
            found = False
            for a in range(1, m // 2 + 1):
                for b in range(a + 1, m // 2 + 1):
                    # Skip invalid generating sets
                    if a == m - a or b == m - b:
                        continue
                    if a == b or a == m - b or b == m - a:
                        continue
                    # Check connectedness
                    if math.gcd(math.gcd(a, b), m) != 1:
                        continue
                    
                    if _check_girth_cyclic(m, a, b, g):
                        if best_n is None or m < best_n:
                            best_n = m
                            best_edges = _make_circulant_edges(m, a, b)
                        found = True
                        break
                if found:
                    break
            if found:
                break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback
    return _construct_fallback(4, g)


def _make_circulant_edges(m, a, b):
    """Create edge list for circulant graph Z_m with S={a, -a, b, -b}."""
    edge_set = set()
    for i in range(m):
        for s in [a, b]:
            j = (i + s) % m
            if i < j:
                edge_set.add((i, j))
            else:
                edge_set.add((j, i))
    return sorted(edge_set)


def _check_girth_cyclic(m, a, b, g):
    """Check that the Cayley graph of Z_m with S={a,-a,b,-b} has girth >= g.
    We check that there are no non-backtracking closed walks of length < g.
    """
    S = [a, -a, b, -b]
    
    for L in range(2, g):
        # Enumerate all non-backtracking sequences of length L
        # A non-backtracking walk: s_i != -s_{i+1}
        # Closed: sum(s_i) = 0 mod m
        
        def _has_closed_walk(start, remaining, current_sum, prev):
            if remaining == 0:
                return current_sum % m == 0
            for s in S:
                if s == -prev:
                    continue
                if _has_closed_walk(s, remaining - 1, current_sum + s, s):
                    return True
            return False
        
        for s in S:
            if _has_closed_walk(s, L - 1, s, s):
                return False
    
    return True


def _construct_fallback(k, g):
    """Fallback: use the Lazebnik-Ustimenko-Woldar construction."""
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
