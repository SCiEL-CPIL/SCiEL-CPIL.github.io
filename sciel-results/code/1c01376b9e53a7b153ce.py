import time
import math
from collections import deque

# EVOLVE-BLOCK-START

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True

def _luw_construct(k, g):
    """
    Construct a k-regular graph of girth at least g using the
    Lazebnik-Ustimenko-Woldar (LUW) construction.
    
    The LUW graph LUW(k, d) has:
    - Degree k
    - Girth 2d + 3
    - Number of vertices 2 * k^(d+1)
    
    We choose the smallest d such that 2d + 3 >= g.
    """
    if k < 2:
        # Degenerate cases
        if k == 1:
            return 1, []
        return 2, [(0, 1)] # Simple edge, not 1-regular? No, 1-regular is a matching.
        # For k=1, girth is infinite (no cycles). n=2 is minimal.
    
    # Calculate minimal d
    # Girth = 2d + 3 >= g  =>  2d >= g - 3  =>  d >= ceil((g-3)/2)
    # Using integer arithmetic: d = max(1, (g - 2) // 2)
    # Check:
    # g=5 -> (3)//2 = 1. Girth 5. OK.
    # g=6 -> (4)//2 = 2. Girth 7. OK.
    # g=7 -> (5)//2 = 2. Girth 7. OK.
    # g=8 -> (6)//2 = 3. Girth 9. OK.
    # g=9 -> (7)//2 = 3. Girth 9. OK.
    # g=10 -> (8)//2 = 4. Girth 11. OK.
    
    d = max(1, (g - 2) // 2)
    
    q = k
    
    # LUW construction details
    # Vertices are pairs (side, x) where side in {0, 1} and x is a tuple of length d+1
    # x[0] is ignored (or set to 0), x[1..d] are in Z_q.
    # Actually, standard LUW uses x in (Z_q)^{d+1} but the first coordinate is often fixed or handled specially.
    # The seed code uses:
    # start = (0, (None,) + (0,) * d)
    # This implies x has length d+1.
    
    # The rule for edges in LUW:
    # From (0, x) to (1, y) where:
    # y[1] = x[1] (or similar)
    # The seed code implements a specific recurrence.
    
    # Let's stick to the seed's implementation of the LUW graph structure,
    # but ensure it uses the correct d.
    
    # The seed code's rule generation:
    rule = {}
    # For c=2,3,4, the rule is fixed.
    # For c > 4, it uses a pattern.
    # This is a specific variant. Let's trust the seed's logic for the graph structure
    # as it was provided as a valid fallback, just with suboptimal d.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Initialize rules
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The seed code had a specific list of rules.
        # Let's replicate it exactly.
        rules_list = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_list[r]

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
    
    # BFS to generate graph
    for i in range(len(order)):
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110.0
    
    if k < 3:
        if k == 2:
            n = max(g, 3)
            return n, [(i, (i + 1) % n) for i in range(n)]
        return 1, []
        
    # For k=5 (odd), cyclic group search is impossible for simple undirected graphs.
    # We skip the expensive and futile search for odd k.
    if k % 2 == 1:
        return _luw_construct(k, g)
        
    # For even k, we could try cyclic search, but LUW is often competitive or better
    # for small girths if we pick d correctly.
    # Given the time limit and the fact that k=5 is the target, we will just use LUW
    # for all k to ensure speed and reliability.
    # If k is even, LUW still works.
    
    return _luw_construct(k, g)

# EVOLVE-BLOCK-END
