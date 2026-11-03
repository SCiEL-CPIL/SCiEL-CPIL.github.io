import time
import sys
import collections

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction with optimized state encoding.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    if k != 3:
        # Fallback for other k values, though task specifies k=3
        return 0, []

    q = k # q=3
    
    def girth_of_d(d):
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6
            
    # Find minimum d such that girth >= g
    d = 3
    while girth_of_d(d) < g:
        d += 1
        
    if time.time() > deadline:
        return 0, []

    # GF(q) operations for q=3
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

    # Rule for LUW
    # rule[c] = (a, b) such that l[c] - p[c] = l[a] * p[b]
    # For c=2: (1,1), c=3: (2,1), c=4: (1,2)
    # For c>=5: cycle through patterns
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
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

    # State encoding:
    # x is a tuple of length d+1 (indices 0 to d).
    # x[0] is ignored (None in original), x[1] is 'first', x[2..d] are computed.
    # We encode x[1..d] as a base-q integer.
    # x[1] is the most significant digit? Or least?
    # Let's use x[1] as the most significant for easy extraction?
    # Actually, the computation depends on x[c] for c up to d.
    # Let's encode the vector (x[1], x[2], ..., x[d]) as an integer.
    # value = x[1] * q^(d-1) + x[2] * q^(d-2) + ... + x[d] * q^0
    
    pow_q = [1] * (d + 1)
    for i in range(1, d + 1):
        pow_q[i] = pow_q[i-1] * q
        
    def encode_x(x_tuple):
        # x_tuple is (None, x1, x2, ..., xd)
        val = 0
        for i in range(1, d + 1):
            val += x_tuple[i] * pow_q[d - i]
        return val

    def decode_x(val):
        # Returns tuple (None, x1, ..., xd)
        x = [0] * (d + 1)
        x[0] = None
        for i in range(1, d + 1):
            x[i] = val % q
            val //= q
        return tuple(x)

    index = {}
    order = []
    edges = []
    
    # Start vertex: side 0, x = (None, 0, ..., 0)
    start_val = 0
    start = (0, start_val)
    index[start] = 0
    order.append(start)
    
    completed = False
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        side, x_val = vertex
        x = decode_x(x_val)
        
        # Generate neighbors
        for first in range(q):
            # y[0] is None, y[1] is 'first', y[2..d] are computed
            y = [0] * (d + 1)
            y[0] = None
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            y_val = encode_x(tuple(y))
            neighbor = (1 - side, y_val)
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            
            # Add edge if side is 0 to avoid duplicates
            if side == 0:
                edges.append((index[vertex], index[neighbor]))
    
    n = len(order)
    
    if time.time() > deadline:
        return 0, []
    
    unique_edges = list(set(edges))
    
    return n, unique_edges
