import time
import random

def _seed_construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
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

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k != 3 or g != 16:
        return _seed_construct(k, g)
        
    target_girth = 16
    max_depth = 7
    
    # Try to find a smaller graph
    for N in [1000, 1024]:
        if time.time() > deadline:
            break
            
        # Generate random 3-regular graph
        adj = generate_random_3_regular(N)
        if adj is None:
            continue
            
        # Local search
        for _ in range(5000):
            if time.time() > deadline:
                break
                
            cycle = find_short_cycle(adj, N, max_depth)
            if cycle is None:
                edges = extract_edges(adj)
                return N, edges
            else:
                adj = break_cycle(adj, cycle, N)
                if adj is None:
                    # Failed to break, restart
                    adj = generate_random_3_regular(N)
                    if adj is None:
                        break
                    
    return _seed_construct(k, g)

def generate_random_3_regular(N):
    # ... implementation ...
    pass

def find_short_cycle(adj, N, max_depth):
    # ... implementation ...
    pass

def break_cycle(adj, cycle, N):
    # ... implementation ...
    pass

def extract_edges(adj):
    # ... implementation ...
    pass
