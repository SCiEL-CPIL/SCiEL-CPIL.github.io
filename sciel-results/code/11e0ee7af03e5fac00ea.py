import numpy as np
import itertools
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    q = k
    # Determine d
    # Girth >= d + 5
    # We need girth >= g
    # So d >= g - 5
    # LUW girth is d+5 for odd d, d+6 for even d?
    # Let's just use d = g - 5 if (g-5)%2==1 else g-4
    # For g=16, d=11.
    
    d = g - 5
    if d % 2 == 0:
        d += 1 # Ensure odd d for girth d+5? Or just use d such that girth >= g.
        
    # Actually, for LUW, girth is at least d+5.
    # So d = g - 5 is sufficient.
    # If g=16, d=11.
    
    # Precompute rules
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
    # Convert to 0-based indices for arrays
    # rule[c] gives (a, b) where a, b are 1-based indices in the vector of length d.
    # In my array x[0..d-1], index i corresponds to 1-based i+1.
    # So 1-based a -> 0-based a-1.
    
    rules_0 = {}
    for c in range(2, d+1):
        a, b = rule[c]
        rules_0[c-1] = (a-1, b-1) # c-1 is the 0-based index of the component to compute
        
    # Total vertices = 2 * q^d
    n_points = q ** d
    n_total = 2 * n_points
    
    edges = []
    
    # Iterate over all points
    # Point x is a tuple of d elements in {0..q-1}
    # ID of point x is sum(x_i * q^i)
    # ID of line y is n_points + sum(y_i * q^i)
    
    powers = [q**i for i in range(d)]
    
    for idx, x in enumerate(itertools.product(range(q), repeat=d)):
        # x is a tuple (x0, ..., x_{d-1})
        # Calculate 3 neighbors
        for first in range(q):
            # y[0] = first
            # y[i] for i=1..d-1
            y = [0] * d
            y[0] = first
            valid = True
            for i in range(1, d):
                # i is 0-based index for y[i]
                # Corresponds to c = i+1 in 1-based
                a_idx, b_idx = rules_0[i]
                # y[i] = x[i] + y[a_idx] * x[b_idx]
                val = (x[i] + y[a_idx] * x[b_idx]) % q
                y[i] = val
            # Convert y to ID
            y_id = n_points + sum(y[i] * powers[i] for i in range(d))
            edges.append((idx, y_id))
            
        if time.time() > deadline:
            break
            
    return n_total, edges
