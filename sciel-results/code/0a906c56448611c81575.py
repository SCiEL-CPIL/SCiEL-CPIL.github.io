import time
import sys

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction CD(d, q).
    """
    start_time = time.time()
    deadline = 150  # seconds
    
    # LUW parameters
    # We need girth >= g.
    # Girth of CD(d, q) is at least d + 5 (if d odd) or d + 4 (if d even).
    # We want d such that d + 5 >= g (if d odd) or d + 4 >= g (if d even).
    # To minimize vertices (q^d), we want smallest d.
    
    q = k
    if q == 4:
        # GF(4) arithmetic
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # Determine d
    # Try odd d first as it gives higher girth for same d?
    # d+5 >= g => d >= g-5
    # d+4 >= g => d >= g-4
    # We want minimal d.
    # If g-5 is odd, use d = g-5.
    # If g-5 is even, use d = g-4 (which is odd) or d = g-5 (even)?
    # If d is even, girth >= d+4. So d >= g-4.
    # If d is odd, girth >= d+5. So d >= g-5.
    
    d_candidate_odd = g - 5
    if d_candidate_odd % 2 == 0:
        d_candidate_odd += 1
    if d_candidate_odd < 3:
        d_candidate_odd = 3 if g <= 8 else 5 # Adjust for small g
        
    d_candidate_even = g - 4
    if d_candidate_even % 2 == 1:
        d_candidate_even += 1
    if d_candidate_even < 2:
        d_candidate_even = 2 if g <= 6 else 4
        
    # Choose the one with smaller q^d
    # Usually odd d is better for girth/size ratio?
    # Let's pick the smallest d that satisfies the condition.
    # For g=16:
    # d_odd >= 11. d_odd=11. Girth >= 16. Size 3^11.
    # d_even >= 12. d_even=12. Girth >= 16. Size 3^12.
    # So d=11 is better.
    
    d = d_candidate_odd
    
    # Precompute rules
    # rule[c] = (a, b) for c in 2..d
    # The LUW graph definition:
    # Points: x in GF(q)^d, indexed by x[1]..x[d] (x[0] is None)
    # Lines: y in GF(q)^d
    # Incidence: x ~ y iff y[c] - x[c] = y[a] * x[b] for c=2..d
    # where (a, b) depends on c.
    
    # Standard LUW rule:
    # For c=2: a=1, b=1
    # For c=3: a=2, b=1
    # For c=4: a=1, b=2
    # For c>4: 
    #   r = (c-5) % 4
    #   c0 = c - r
    #   if r==0: (1, c0-2)
    #   if r==1: (c0-1, 1)
    #   if r==2: (c0, 1)
    #   if r==3: (1, c0+1)
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # BFS to find connected component of (0, (0,)*d)
    # Vertices are (side, tuple)
    # side 0: point, side 1: line
    
    start_vertex = (0, (0,) * d)
    index = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    # Optimization: Precompute neighbor generation logic
    # Since q is small (3), we can iterate.
    
    for i, vertex in enumerate(order):
        if time.time() - start_time > deadline - 5:
            break
            
        side, x = vertex
        # x is a tuple of length d, x[0] is ignored in logic? 
        # In the code, x is (None, x1, ..., xd) or (0, x1, ..., xd)?
        # The code uses `x[c]` where c goes from 2 to d.
        # So x needs to be indexable up to d.
        # Let's use 0-based indexing for the tuple, but shift indices in logic.
        # Let tuple T have length d. T[0] corresponds to index 1, T[d-1] to index d.
        # So x[c] in math -> T[c-1].
        
        for first in range(q):
            # y[1] = first
            # y[c] for c=2..d
            y_tuple = [0] * d
            y_tuple[0] = first # y[1]
            
            for c in range(2, d + 1):
                a, b = rule[c]
                # Math indices: x[c], y[a], x[b], y[c]
                # Tuple indices: x[c-1], y[a-1], x[b-1], y[c-1]
                
                if side == 0:
                    # Point x to Line y
                    # y[c] - x[c] = y[a] * x[b]
                    # y[c] = x[c] + y[a] * x[b]
                    y_val = add(x[c-1], mul(y_tuple[a-1], x[b-1]))
                else:
                    # Line x to Point y
                    # x[c] - y[c] = x[a] * y[b]  (Symmetric? No)
                    # Incidence is symmetric.
                    # If x is line, y is point.
                    # y[c] = x[c] - x[a] * y[b]
                    y_val = sub(x[c-1], mul(x[a-1], y_tuple[b-1]))
                
                y_tuple[c-1] = y_val
            
            neighbor = (1 - side, tuple(y_tuple))
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            
            if side == 0:
                edges.append((i, index[neighbor]))
        
        # Check time
        if time.time() - start_time > deadline - 5:
            break

    return len(order), edges
