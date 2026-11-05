import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    q = k
    best_result = None
    
    # Try multiple d values and rule variations
    for d in range(max(3, g - 6), g + 2):
        if time.time() > deadline - 5:
            break
        
        # Try different rule generation patterns
        for pattern in range(3):
            rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d + 1):
                if pattern == 0:
                    r, c0 = (c - 5) % 4, c - (c - 5) % 4
                    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
                elif pattern == 1:
                    r, c0 = (c - 5) % 3, c - (c - 5) % 3
                    rule[c] = [(1, c0 - 1), (c0, 1), (1, c0 + 1)][r]
                else:
                    r, c0 = (c - 5) % 5, c - (c - 5) % 5
                    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1), (c0 + 2, 1)][r % 5]
            
            if q == 4:
                table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
                add_op = lambda x, y: x ^ y
                sub_op = lambda x, y: x ^ y
                mul_op = lambda x, y: table[x][y]
            else:
                add_op = lambda x, y: (x + y) % q
                sub_op = lambda x, y: (x - y) % q
                mul_op = lambda x, y: x * y % q
            
            def neighbours(vertex, rule=rule, add_op=add_op, sub_op=sub_op, mul_op=mul_op, q=q, d=d):
                side, x = vertex
                for first in range(q):
                    y = [None, first] + [0] * (d - 1)
                    for c in range(2, d + 1):
                        a, b = rule[c]
                        y[c] = add_op(x[c], mul_op(y[a], x[b])) if side == 0 else sub_op(x[c], mul_op(x[a], y[b]))
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
            
            n = len(order)
            
            # Quick check: verify it's k-regular
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            valid = all(len(adj[i]) == k for i in range(n))
            
            if valid and (best_result is None or n < best_result[0]):
                best_result = (n, edges)
    
    if best_result is not None:
        return best_result
    
    # Fallback
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
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
