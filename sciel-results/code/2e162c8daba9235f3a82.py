"""Cayley-graph construction for 4-regular girth-10 graph."""
import itertools
import numpy as np

def construct(k, g):
    # We construct a 4-regular graph on a finite abelian group Z_m^d (or similar)
    # using a symmetric 4-element generating set, and verify girth >= g.
    # For k=4, g=10, we search over small abelian groups.
    
    if k != 4:
        # Fallback: use a simple construction for other k
        # For now, handle k=4 specifically
        return _construct_fallback(k, g)
    
    # Strategy: try groups Z_m (cyclic) and Z_a x Z_b
    # A 4-regular Cayley graph on a group G with generating set S = {s1, s2, s1^-1, s2^-1}
    # (where s1 != s2^-1, s2 != s1^-1, etc.)
    
    # For girth >= 10, we need no closed non-backtracking walks of length < 10.
    # This means:
    # - No relations of length 2, 3, 4, 5, 6, 7, 8, 9 in the non-backtracking sense.
    
    # Let's try cyclic groups Z_m with S = {1, -1, a, -a} for some a.
    # Or Z_m with S = {a, b, -a, -b}.
    
    best_n = None
    best_edges = None
    
    # Try cyclic groups Z_m for m from some range
    # The Moore bound for (4,10) is 243, so n >= 243.
    # Try m from 243 to ~400
    
    for m in range(243, 420):
        # Try generating sets {a, b, -a, -b} with 1 <= a < b < m/2
        # and gcd(a, b, m) = 1 (for connectedness)
        found = False
        for a in range(1, m // 2 + 1):
            for b in range(a + 1, m // 2 + 1):
                if a == m - a or b == m - b:
                    continue  # would create 2-cycles
                if a == b or a == m - b or b == m - a:
                    continue
                # Check connectedness: gcd(a, b, m) == 1
                import math
                if math.gcd(math.gcd(a, b), m) != 1:
                    continue
                
                # Verify girth >= 10 by checking non-backtracking closed walks
                if _check_girth_cyclic(m, a, b, g):
                    # Construct edges
                    edges = []
                    for i in range(m):
                        for s in [a, b]:
                            j = (i + s) % m
                            if i < j:
                                edges.append((i, j))
                    # Add the -a and -b edges (they are the same as a and b edges in undirected)
                    # Actually, for undirected Cayley graph with S = {a, -a, b, -b},
                    # edges are (i, i+a) and (i, i+b) for all i, but we only add i<j to avoid duplicates
                    # Wait, (i, i+a) and (i+a, i) are the same edge. So we add each edge once.
                    # Let me recount: for each i, edges to i+a and i+b.
                    # Total edges = m * 2 = 2m. Degree = 4. Correct.
                    
                    # But we need to make sure we're not adding duplicate edges
                    # Edge (i, i+a) is the same as (i+a, i). So we should only add when i < i+a mod m?
                    # No, in a cyclic group, we just add all (i, (i+s)%m) for s in {a,b} and i in 0..m-1
                    # but this gives 2m edges, each appearing twice if we're not careful.
                    # Actually, (i, (i+a)%m) is an undirected edge. The edge {(i+a)%m, i} is the same.
                    # So if we iterate i from 0 to m-1 and add (i, (i+a)%m), we get each edge twice
                    # unless (i+a)%m == i, which can't happen.
                    # Wait no: edge (i, j) where j=(i+a)%m. When i' = j, we'd add (j, (j+a)%m) which is different.
                    # The same undirected edge {i, j} is added when i is the smaller? No.
                    # Let me just use a set.
                    
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


def _check_girth_cyclic(m, a, b, g):
    """Check that the Cayley graph of Z_m with S={a,-a,b,-b} has girth >= g.
    We check that there are no non-backtracking closed walks of length < g.
    """
    # A non-backtracking walk of length L is a sequence s1, s2, ..., sL from {a, -a, b, -b}
    # such that si != -s_{i+1} for all i (no immediate backtracking),
    # and sum(s1+...+sL) = 0 mod m.
    
    # For small L, we can enumerate.
    # L = 2: s1 + s2 = 0, s2 = -s1. But s2 != -s1 is violated. So no length-2 non-backtracking closed walks.
    # L = 3: s1+s2+s3=0, s2!=-s1, s3!=-s2. 
    #   Possibilities: (a, a, -2a), (a, b, -(a+b)), etc.
    #   We need to check if any such triple sums to 0 mod m with the non-backtracking constraint.
    
    S = [a, -a, b, -b]
    
    for L in range(2, g):
        # Enumerate all non-backtracking sequences of length L
        # This is 4 * 3^(L-1) possibilities
        # For L up to 9, that's 4 * 3^8 = 26244, which is fine.
        
        def _enumerate(start, length, current_sum, prev):
            if length == 0:
                return current_sum % m == 0
            for s in S:
                if s == -prev:
                    continue
                if _enumerate(s, length - 1, current_sum + s, s):
                    return True
            return False
        
        for s in S:
            if _enumerate(s, L - 1, s, s):
                return False
    
    return True


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
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
