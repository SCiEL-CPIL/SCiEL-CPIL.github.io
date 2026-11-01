import time

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    if k == 4:
        q = 4
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
        mul_table = [[0, 0, 0, 0], 
                     [0, 1, 2, 3], 
                     [0, 2, 3, 1], 
                     [0, 3, 1, 2]]
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def get_val(packed, idx):
            return (packed >> (idx * 2)) & 3

        def gen_neighbours(side, packed):
            for first in range(q):
                y_vals = [0] * d
                y_vals[0] = first
                for c in range(2, d + 1):
                    a, b = rule[c]
                    idx_c = c - 1
                    idx_a = a - 1
                    idx_b = b - 1
                    x_c = get_val(packed, idx_c)
                    if side == 0:
                        ya = y_vals[idx_a]
                        xb = get_val(packed, idx_b)
                        prod = mul_table[ya][xb]
                        y_vals[idx_c] = (x_c + prod) & 3
                    else:
                        xa = get_val(packed, idx_a)
                        yb = y_vals[idx_b]
                        prod = mul_table[xa][yb]
                        y_vals[idx_c] = (x_c - prod) & 3
                y_packed = 0
                for i in range(d):
                    y_packed |= (y_vals[i] << (i * 2))
                yield 1 - side, y_packed

        start_vertex = (0, 0)
        index = {start_vertex: 0}
        order = [start_vertex]
        edges = []
        
        for i in range(len(order)):
            if time.time() > deadline:
                break
            v = order[i]
            side, packed = v
            for ns, np in gen_neighbours(side, packed):
                key = (ns, np)
                if key not in index:
                    index[key] = len(order)
                    order.append(key)
                if side == 0:
                    edges.append((index[v], index[key]))
        return len(order), edges

    else:
        # Fallback for other k (less optimized)
        # Use seed logic with time guard
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
        index = {start: 0}
        order = [start]
        edges = []
        for i in range(len(order)):
            if time.time() > deadline:
                break
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
# EVOLVE-BLOCK-END
