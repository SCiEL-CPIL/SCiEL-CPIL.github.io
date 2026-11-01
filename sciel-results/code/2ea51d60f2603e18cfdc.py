import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def _known_cages(k, g):
        """Return (n, edges) for known small cages, or None."""
        # Petersen graph: 3-regular, girth 5, n=10
        if k == 3 and g == 5:
            edges = [(0,1),(1,2),(2,3),(3,4),(4,0),
                     (5,7),(7,9),(9,6),(6,8),(8,5),
                     (0,5),(1,6),(2,7),(3,8),(4,9)]
            return 10, edges
        
        # Heawood graph: 3-regular, girth 7, n=14
        if k == 3 and g == 7:
            # Incidence graph of Fano plane
            edges = []
            # Points 0-6, Lines 7-13
            # Lines: 012, 034, 056, 135, 146, 236, 245
            lines = [(0,1,2),(0,3,4),(0,5,6),(1,3,5),(1,4,6),(2,3,6),(2,4,5)]
            for i, line in enumerate(lines):
                for pt in line:
                    edges.append((pt, 7 + i))
            return 14, edges
        
        # (4,5)-cage: n=20
        # The (4,5) cage is the 4-regular graph on 20 vertices with girth 5.
        # It can be constructed from the 5-dimensional hypercube Q5? No.
        # It is the complement of the union of 4 disjoint 5-cycles?
        # Actually, the (4,5) cage is the graph where vertices are pairs (i, j) with i,j in Z5, 
        # and edges connect (i,j) to (i+1, j), (i-1, j), (i, j+1), (i, j-1)?? 
        # That's the grid, which has girth 4.
        # The (4,5) cage is known to be the 20-vertex graph.
        # Construction: Take two 5-cycles. Add edges between them.
        # Actually, I will rely on LUW for (4,5) if I can't hardcode it.
        # LUW q=4, d=2 gives n=32.
        # Known best is 20.
        # I don't have the edges for 20.
        
        # (5,5)-cage: n=30
        # LUW q=5, d=2 gives n=50.
        # Known best is 30.
        
        return None

    def luw_construct(k, g):
        """
        Construct a Lepton-Upper-Water graph.
        Parameters:
        k: degree
        g: girth (must be odd for standard LUW? No, LUW works for g = 2d+1)
        """
        if k < 2:
            return None
            
        # LUW requires q = k (for k-regular)
        # girth g = 2d + 1 => d = (g - 1) / 2
        if (g - 1) % 2 != 0:
            # If g is even, LUW standard form might need adjustment or we use d = (g-2)/2?
            # Standard LUW gives girth at least 2d+1.
            # If we want girth >= g, we need 2d+1 >= g => d >= (g-1)/2.
            # So d = ceil((g-1)/2).
            d = (g - 1 + 1) // 2 
        else:
            d = (g - 1) // 2
            
        # If d is too small, LUW might not work or girth might be lower?
        # LUW requires d >= 3 for the rule table used here?
        # The rule table in the seed was defined for d up to d.
        # Let's ensure d is at least 2 or 3.
        if d < 2:
            # For g=3, d=1. LUW with d=1?
            # Girth 3 triangles.
            # LUW q=k, d=1.
            # Neighbors: (0, x) -> (1, y) where y[1] = x[1] + a*x[b]...
            # For d=1, the rule is just a multiplication/addition.
            # Let's handle d=1 separately or force d=2 and check girth?
            # If g=3, we just want any k-regular graph.
            # Cycle C_k? No, k-regular.
            # Complete graph K_{k+1} is k-regular, girth 3.
            # n = k+1.
            if g <= 3:
                n = k + 1
                edges = [(i, j) for i in range(n) for j in range(i+1, n)]
                return n, edges
            # If g=4, d=1.5 -> d=2?
            # If g=4, we need girth >= 4.
            # LUW with d=1 gives girth >= 3.
            # LUW with d=2 gives girth >= 5.
            # So for g=4, we can use d=2 (girth >= 5 is fine) or find a specific graph.
            # Using d=2 is safer for girth, but n is larger.
            # n = 2*k^2.
            # For k=3, g=4. n=18.
            # Is there a 3-regular graph of girth 4 with n<18?
            # C_6 is 2-reg.
            # Cube Q_3 is 3-reg, girth 4, n=8.
            # So for g=4, LUW d=2 is suboptimal.
            # However, the task focuses on large g.
            pass

        q = k
        # Ensure q is prime or 4 for simple field operations
        # If k is not prime and not 4, LUW with standard field ops fails.
        # We can only use LUW if k is prime or k=4.
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        if q not in primes and q != 4:
            # Fallback for non-prime k (e.g., k=6, 8, 10)
            # We cannot use LUW directly.
            # We might return a trivial graph or raise an error.
            # For the purpose of this task (k=5), this path is not taken.
            return None

        # Define field operations
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return x * y % q

        # Define the LUW rule
        # rule[c] = (a, b) for c in 2..d
        # The rule determines how to compute y[c] from x and previous y's.
        # Standard LUW rule:
        # For c=2: (1, 1) -> y[2] = x[2] + y[1]*x[1]
        # For c=3: (2, 1) -> y[3] = x[3] + y[2]*x[1]
        # For c=4: (1, 2) -> y[4] = x[4] + y[1]*x[2]
        # For c=5: (1, 3) ?
        # The seed used:
        # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        # for c in range(5, d+1):
        #   r, c0 = (c-5)%4, c - (c-5)%4
        #   rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
        
        # This rule set is specific. I will use it.
        
        rule = {}
        if d >= 2: rule[2] = (1, 1)
        if d >= 3: rule[3] = (2, 1)
        if d >= 4: rule[4] = (1, 2)
        
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            # Indices:
            # r=0: (1, c0-2)
            # r=1: (c0-1, 1)
            # r=2: (c0, 1)
            # r=3: (1, c0+1)
            if r == 0:
                rule[c] = (1, c0 - 2)
            elif r == 1:
                rule[c] = (c0 - 1, 1)
            elif r == 2:
                rule[c] = (c0, 1)
            else:
                rule[c] = (1, c0 + 1)

        def neighbours(vertex):
            side, x = vertex
            # x is a tuple of length d+1. x[0] is None or 0?
            # Seed: start = (0, (None,) + (0,) * d)
            # So x[0] is None. x[1..d] are values.
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                # y[0] is None, y[1] is 'first'
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        # y[c] = x[c] + y[a] * x[b]
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        # y[c] = x[c] - x[a] * y[b]
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        # BFS to generate graph
        for i in range(len(order)):
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                # Add edge only once (from side 0 to side 1)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    # 1. Try known cages
    known = _known_cages(k, g)
    if known is not None:
        return known

    # 2. Try LUW
    luw_result = luw_construct(k, g)
    if luw_result is not None:
        return luw_result

    # 3. Fallback: Simple cycle or complete graph if LUW fails
    # For k=2, g can be anything? No, k=2 is cycle.
    if k == 2:
        # Cycle C_g has girth g. n=g.
        # But must be 2-regular.
        # If g is small, n=g.
        # If g is large, n=g.
        edges = [(i, (i+1) % g) for i in range(g)]
        return g, edges

    # If we get here, we failed to construct.
    # This should not happen for k=5, g=9.
    raise ValueError(f"Cannot construct {k}-regular graph of girth {g}")

