import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = 115.0  # 5s margin
    
    # Moore Bound Lower Bound
    if g % 2 == 0:
        r = g // 2
        lb = 1 + k * sum([(k-1)**i for i in range(r)])
    else:
        r = (g - 1) // 2
        lb = 2 * sum([(k-1)**i for i in range(r)])
        
    # For very small cases, try to find the exact cage via backtracking
    # This is only feasible for small n.
    # We try n from lb upwards.
    
    # If lb is small, we can search.
    # Let's set a hard cap on n for backtracking to ensure we don't hang.
    # If lb > 60, we skip backtracking and go to LUW.
    
    if lb <= 60:
        # Try to find a solution with n = lb, lb+1, ... up to a limit
        # or until we find one.
        # We use a simple backtracking algorithm.
        # State: current graph edges, degrees of vertices.
        
        # To make it faster, we can use the known cages for (3,5), (3,7), etc.
        # But a generic solver is better.
        
        # Given the complexity, and that LUW is valid, we only use backtracking
        # if we are confident we can finish quickly.
        # For k=3, g=5, lb=10. Solution is Petersen.
        # For k=3, g=7, lb=24. Solution is Tutte-Coxeter.
        
        # We will attempt a limited search.
        # If we don't find a solution quickly, we fallback to LUW.
        
        # Actually, implementing a robust backtracker for regular graphs of high girth
        # is complex. A simpler heuristic:
        # If we know the optimal n for small (k,g), use it.
        # Known cages:
        # (3,5): 10
        # (3,7): 24
        # (3,9): 70
        # (4,5): 19? No, (4,5) is 19? Moore bound 1+4+4*3=17. (4,5)-cage is 19? 
        # Actually, (4,5) cage is the **Foster** graph? No.
        
        # Let's just use LUW for all cases to be safe, unless we can verify a smaller n.
        # The LUW construction is valid.
        
        pass

    # Use LUW Construction
    return _luw_construct(k, g)

def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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
