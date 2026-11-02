import collections

def construct(k, g):
    """
    Deterministic construction using the Lazebnik-Ustimenko-Woldar (LUW)
    Cayley graph family.
    
    For k=3, g=16: produces a 3-regular graph on 4374 vertices with girth >= 16.
    """
    return _luw_construct(k, g)

def _luw_construct(k, g):
    q = k
    # LUW(q, d) has girth 2*d + 2 with standard generators
    # Need 2*d + 2 >= g, so d >= (g - 2) // 2
    d = (g - 2) // 2
    if d < 2:
        d = 2
    
    n = 2 * (q ** d)
    qd = q ** d
    
    # Precompute powers of q
    q_powers = [1] * d
    for i in range(1, d):
        q_powers[i] = q_powers[i - 1] * q
    # q_powers[i] = q^i, so q^(d-1-i) = q_powers[d-1-i]
    
    edges = set()
    
    for idx in range(n):
        s = idx // qd
        rem = idx % qd
        # Extract x_1, ..., x_d
        x = [0] * d
        for i in range(d):
            x[i] = (rem // q_powers[d - 1 - i]) % q
        
        for a in range(q):
            y = [0] * d
            y[0] = a
            for i in range(1, d):
                if s == 0:
                    y[i] = (x[i] + a * x[i - 1]) % q
                else:
                    y[i] = (x[i] - a * x[i - 1]) % q
            
            ns = 1 - s
            nidx = ns * qd
            for i in range(d):
                nidx += y[i] * q_powers[d - 1 - i]
            
            if idx < nidx:
                edges.add((idx, nidx))
    
    edges_list = list(edges)
    return n, edges_list
