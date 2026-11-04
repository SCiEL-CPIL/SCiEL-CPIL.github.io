import time
import itertools

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def luw_construct(k, g):
        q = k
        d = max(3, (g - 2) // 2)
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

    if k != 3:
        return luw_construct(k, g)

    def check_cayley_graph(q, g):
        if time.time() > deadline - 10:
            return None
            
        def normalize(m):
            a, b, c, d = m
            if a < 0:
                return (-a, -b, -c, -d)
            if a == 0 and b < 0:
                return (-a, -b, -c, -d)
            if a == 0 and b == 0 and c < 0:
                return (-a, -b, -c, -d)
            return m

        def mul(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            a3 = (a1*a2 + b1*c2) % q
            b3 = (a1*b2 + b1*d2) % q
            c3 = (c1*a2 + d1*c2) % q
            d3 = (c1*b2 + d1*d2) % q
            return normalize((a3, b3, c3, d3))

        identity = normalize((1, 0, 0, 1))
        
        invs = []
        seen = set()
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    if a != 0:
                        try:
                            a_inv = pow(a, q-2, q)
                        except:
                            continue
                        d = (1 + b*c) * a_inv % q
                        m = normalize((a, b, c, d))
                        if m in seen:
                            continue
                        seen.add(m)
                        m2 = mul(m, m)
                        if m2 == identity:
                            invs.append(m)
                    else:
                        if b == 0:
                            continue
                        try:
                            b_inv = pow(b, q-2, q)
                        except:
                            continue
                        c_val = (q-1) * b_inv % q
                        for d in range(q):
                            m = normalize((0, b, c_val, d))
                            if m in seen:
                                continue
                            seen.add(m)
                            m2 = mul(m, m)
                            if m2 == identity:
                                invs.append(m)
        
        if len(invs) < 3:
            return None
            
        group_size = q * (q*q - 1) // 2
        
        for a_idx in range(len(invs)):
            if time.time() > deadline - 5:
                return None
            a = invs[a_idx]
            for b_idx in range(a_idx + 1, len(invs)):
                b = invs[b_idx]
                for c_idx in range(b_idx + 1, len(invs)):
                    c = invs[c_idx]
                    gens = [a, b, c]
                    
                    queue = [(identity, None, 0)]
                    visited = set()
                    visited.add((identity, None))
                    found_cycle = False
                    valid = True
                    
                    while queue:
                        u, v, dist = queue.pop(0)
                        for gen in gens:
                            next_u = mul(u, gen)
                            if next_u == v:
                                continue
                            
                            if next_u == identity:
                                if dist + 1 < g:
                                    valid = False
                                    break
                                else:
                                    # Shortest cycle is >= g
                                    # We can stop searching for this triple
                                    break
                            
                            state = (next_u, u)
                            if state not in visited:
                                visited.add(state)
                                queue.append((next_u, u, dist + 1))
                        
                        if not valid:
                            break
                    
                    if not valid:
                        continue
                        
                    if len(set(e for e, _ in visited)) < group_size:
                        continue
                    
                    elements = list(set(e for e, _ in visited))
                    elem_to_idx = {e: i for i, e in enumerate(elements)}
                    edges = []
                    for e in elements:
                        i = elem_to_idx[e]
                        for gen in gens:
                            next_e = mul(e, gen)
                            j = elem_to_idx[next_e]
                            if i < j:
                                edges.append((i, j))
                    
                    return len(elements), edges
        return None

    for q in [11, 13, 17]:
        result = check_cayley_graph(q, g)
        if result:
            return result
            
    return luw_construct(k, g)
