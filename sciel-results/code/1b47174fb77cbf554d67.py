import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # LUW parameters
    # For k=3, q must be 3.
    # Girth g = d + 5 (odd d) or d + 6 (even d)
    # Find smallest d such that girth >= g
    
    def girth_of_d(d):
        return d + 5 if d % 2 == 1 else d + 6

    d = 3
    while girth_of_d(d) < g:
        d += 1
        
    q = k # q=3
    
    # GF(q) operations
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

    # Rule generation
    # Standard LUW rules for q=3
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]

    n = 2 * (q ** d)
    edges = []
    
    # Direct enumeration
    # Point index: 0 to q^d - 1
    # Line index: q^d to 2*q^d - 1
    
    q_d = q ** d
    
    for pt_idx in range(q_d):
        # Decode point coordinates x[1]..x[d]
        x = [0] * (d + 1)
        tmp = pt_idx
        for j in range(d, 0, -1):
            x[j] = tmp % q
            tmp //= q
        
        # Generate neighbors (lines)
        for y1 in range(q):
            y = [0] * (d + 1)
            y[1] = y1
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b]))
            
            # Encode line index
            ln_idx = q_d
            tmp = 0
            for j in range(1, d + 1):
                tmp = tmp * q + y[j]
            ln_idx += tmp
            
            edges.append((pt_idx, ln_idx))
            
        if time.time() > deadline:
            # If timeout, we have to return something valid.
            # But LUW is fast.
            break

    return n, edges
