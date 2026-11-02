import math
import time

def construct(k, g):
    # Moore bound for k=3, even g
    # M = 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
    r = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(r))
    
    # Ensure moore_bound is even for circulant 3-regular (needs n/2)
    if moore_bound % 2 != 0:
        moore_bound += 1
        
    # Upper bound for search to avoid infinite loop
    # The LUW construction gives a valid upper bound
    # For k=3, g=16, LUW gives ~350k. We can set a lower limit like 10000 for safety
    # But we expect to find a small one.
    max_n = 5000
    
    start_time = time.time()
    
    for n in range(moore_bound, max_n + 1, 2):
        if time.time() - start_time > 110:
            # Fallback to LUW if we run out of time
            return _luw_fallback(k, g)
            
        # Try all valid 'a'
        for a in range(1, n // 2):
            # S = {a, n-a, n/2}
            # Check connectivity: gcd(n, a, n/2) == 1
            if math.gcd(n, a, n // 2) != 1:
                continue
                
            S = [a, n - a, n // 2]
            
            # Check girth
            # R_d is the set of sums of d steps
            R_prev = {0}
            valid = True
            for d in range(1, g):
                R_curr = set()
                for x in R_prev:
                    for s in S:
                        R_curr.add((x + s) % n)
                if 0 in R_curr:
                    valid = False
                    break
                R_prev = R_curr
                # Optimization: if R_prev becomes all of Z_n, and 0 is not in it, 
                # it will never be in it? No, 0 is in Z_n. 
                # If R_prev == set(range(n)), then 0 is in R_prev. 
                # So if 0 is not in R_prev, R_prev != Z_n.
                
            if valid:
                # Construct edges
                edges = []
                for u in range(n):
                    for s in S:
                        v = (u + s) % n
                        if u < v:
                            edges.append((u, v))
                return n, edges

    # If no small n found, return LUW fallback
    return _luw_fallback(k, g)

def _luw_fallback(k, g):
    # The original LUW construction
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
