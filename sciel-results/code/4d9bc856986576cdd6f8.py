import numpy as np
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k == 4 and g == 10:
        q = 11
        
        # Generate SL(2, q) elements
        # A = [[a, b], [c, d]] with ad - bc = 1
        sl2_elements = []
        seen_sl2 = set()
        
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    if a == 0:
                        # det = -bc = 1 => bc = q-1
                        if b == 0:
                            continue
                        # c = (q-1) * b^-1
                        b_inv = pow(b, q-2, q)
                        c_val = (q - 1) * b_inv % q
                        d_val = 0
                        m = (a, b, c_val, d_val)
                        if m not in seen_sl2:
                            seen_sl2.add(m)
                            sl2_elements.append(m)
                    else:
                        a_inv = pow(a, q-2, q)
                        d_val = (1 + b * c) % q
                        d_val = d_val * a_inv % q
                        m = (a, b, c, d_val)
                        # Verify det
                        if (a * d_val - b * c) % q == 1:
                            if m not in seen_sl2:
                                seen_sl2.add(m)
                                sl2_elements.append(m)
        
        # Map to PSL(2, q)
        # Canonical representative: 
        # 1. If a != 0, make a positive (1..q-1)
        # 2. If a == 0 and b != 0, make b positive
        # 3. If a == 0 and b == 0, make c positive
        
        def canonical(m):
            a, b, c, d = m
            # Check if this is the "positive" one
            # Compare with -m = (q-a, q-b, q-c, q-d)
            neg_m = ((q - a) % q, (q - b) % q, (q - c) % q, (q - d) % q)
            
            # Priority 1: a > 0
            if a != 0:
                if a < (q - a) % q:
                    return m
                else:
                    return neg_m
            # Priority 2: a == 0, b > 0
            if b != 0:
                if b < (q - b) % q:
                    return m
                else:
                    return neg_m
            # Priority 3: a == 0, b == 0, c > 0
            # c must be non-zero because det=1
            if c < (q - c) % q:
                return m
            else:
                return neg_m

        psl2_elements = []
        seen_psl2 = set()
        
        for m in sl2_elements:
            cm = canonical(m)
            if cm not in seen_psl2:
                seen_psl2.add(cm)
                psl2_elements.append(cm)
        
        n = len(psl2_elements)
        index_map = {m: i for i, m in enumerate(psl2_elements)}
        
        def mat_mul(x, y):
            a1, b1, c1, d1 = x
            a2, b2, c2, d2 = y
            return ((a1*a2 + b1*c2) % q,
                    (a1*b2 + b1*d2) % q,
                    (c1*a2 + d1*c2) % q,
                    (c1*b2 + d1*d2) % q)
        
        # Generators for 4-regular graph
        u = (1, 1, 0, 1)
        v = (1, 0, 1, 1)
        u_inv = (1, q-1, 0, 1)
        v_inv = (1, 0, q-1, 1)
        
        gens = [u, u_inv, v, v_inv]
        
        edges = []
        seen_edges = set()
        
        for i, m in enumerate(psl2_elements):
            for gen in gens:
                # Calculate neighbor in SL(2, q)
                prod = mat_mul(m, gen)
                # Map to PSL(2, q)
                prod_canon = canonical(prod)
                
                if prod_canon in index_map:
                    j = index_map[prod_canon]
                    if i < j:
                        edge = (i, j)
                        if edge not in seen_edges:
                            seen_edges.add(edge)
                            edges.append(edge)
        
        return n, edges
    
    # Fallback: Lazard-Ustimenko (same as seed)
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
