import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin

    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    # Girth is 2d + 2. We need 2d + 2 >= g => d >= (g-2)//2
    q = k
    d = max(3, (g - 2) // 2)
    
    # If q is not a prime power or 4, LUW might not apply directly or might be invalid.
    # The seed program handles q=4 and prime q.
    # For this task, k=4, so q=4.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    elif q in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]: # Small primes
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    else:
        # Fallback to seed logic or raise error? 
        # For k=4, q=4 is handled.
        # If k is not 4 or prime, we might fail. 
        # But the task is specifically for k=4, g=10.
        # We assume k=4.
        pass

    # Rule for LUW graph
    # rule[c] = (a, b) such that y[c] = x[c] + y[a] * x[b]
    # For c=2, rule is (1,1) ? 
    # The seed program has a specific rule generation.
    # I will copy the rule generation from the seed program to be safe.
    
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
