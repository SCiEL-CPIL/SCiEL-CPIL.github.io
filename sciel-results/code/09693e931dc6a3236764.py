import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    # Moore bound
    if g % 2 == 0:
        half_g = g // 2
        moore_bound = 2 * sum((k - 1) ** i for i in range(half_g))
    else:
        half_g = (g - 3) // 2
        moore_bound = k * sum((k - 1) ** i for i in range(half_g + 1))
        
    # Try Circulant Search for small k
    if k <= 7:
        max_n = 6000
        trials_per_n = 100
        
        for n in range(moore_bound, max_n + 1):
            if time.time() > deadline - 10:
                break
            
            # Determine generator structure
            # k=3: +/-a, b (2b!=0)
            # k=4: +/-a, +/-b
            # k=5: +/-a, +/-b, c (2c!=0)
            # k=6: +/-a, +/-b, +/-c
            # k=7: +/-a, +/-b, +/-c, d (2d!=0)
            
            found = False
            for _ in range(trials_per_n):
                if time.time() > deadline - 5:
                    break
                
                valid_gen = True
                generators = []
                neighbors_list = [] # List of lambda u: v
                
                if k == 3:
                    a = random.randint(1, n // 2)
                    b = random.randint(1, n - 1)
                    if (2 * b) % n == 0: continue
                    if a == b or a == n - b: continue
                    generators = [a, -a, b]
                elif k == 4:
                    a = random.randint(1, n // 2)
                    b = random.randint(1, n // 2)
                    if a == b or a == n - b: continue
                    generators = [a, -a, b, -b]
                elif k == 5:
                    a = random.randint(1, n // 2)
                    b = random.randint(1, n // 2)
                    c = random.randint(1, n - 1)
                    if (2 * c) % n == 0: continue
                    if a == b or a == n - b: continue
                    if a == c or a == n - c or b == c or b == n - c: continue
                    generators = [a, -a, b, -b, c]
                elif k == 6:
                    a = random.randint(1, n // 2)
                    b = random.randint(1, n // 2)
                    c = random.randint(1, n // 2)
                    if a == b or a == n - b or a == c or a == n - c or b == c or b == n - c: continue
                    generators = [a, -a, b, -b, c, -c]
                elif k == 7:
                    a = random.randint(1, n // 2)
                    b = random.randint(1, n // 2)
                    c = random.randint(1, n // 2)
                    d = random.randint(1, n - 1)
                    if (2 * d) % n == 0: continue
                    # Check distinctness
                    gens_set = {a, n-a, b, n-b, c, n-c, d}
                    if len(gens_set) < 7: continue
                    generators = [a, -a, b, -b, c, -c, d]
                else:
                    break # Fallback for k > 7
                
                # Check Girth via BFS
                visited = {0: 0}
                queue = [(0, 0)]
                valid = True
                while queue:
                    u, depth = queue.pop(0)
                    if depth >= g - 1:
                        continue
                    for gen in generators:
                        v = (u + gen) % n
                        if v == 0 and depth + 1 < g:
                            valid = False
                            break
                        if v not in visited:
                            visited[v] = depth + 1
                            queue.append((v, depth + 1))
                    if not valid:
                        break
                
                if valid:
                    edges = []
                    for u in range(n):
                        for gen in generators:
                            v = (u + gen) % n
                            if u < v:
                                edges.append((u, v))
                    return n, edges
                    found = True
                    break
            if found:
                break

    # Fallback: LUW Construction
    # ... (Implement LUW as in seed) ...
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
