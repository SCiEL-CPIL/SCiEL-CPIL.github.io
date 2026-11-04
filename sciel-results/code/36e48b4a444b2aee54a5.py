import random
import time

def construct(k, g):
    # If the specific target is not (3, 16), fall back to the original LUW construction
    if k != 3 or g != 16:
        return _construct_luw(k, g)
    
    # Try to find a small circulant (3, 16)-cage
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Moore bound for (3, 16) is 512
    n_min = 512
    n_max = 1500
    
    # We search for n in this range
    for n in range(n_min, n_max + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a number of random pairs (b, c) for this n
        # We fix a = 1
        # We need 1, b, c to be distinct and not inverses
        # b, c in [1, n-1]
        # To avoid duplicates, we can restrict b, c to [1, n//2] roughly, 
        # but careful with inverses.
        
        trials = 0
        max_trials = 2000  # Adjust based on speed
        
        while trials < max_trials:
            if time.time() - start_time > time_limit:
                break
            trials += 1
            
            b = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
            
            # Check distinctness and non-inverses
            # Generators: 1, b, c
            # Directions: 1, -1, b, -b, c, -c
            # All must be distinct mod n
            dirs = [1, -1, b, -b, c, -c]
            if len(set(d % n for d in dirs)) < 6:
                continue
                
            # Check girth
            if _check_girth_circulant(n, [1, b, c], g):
                # Construct edges
                edges = []
                seen = set()
                for i in range(n):
                    for a in [1, b, c]:
                        j = (i + a) % n
                        u, v = min(i, j), max(i, j)
                        if (u, v) not in seen:
                            seen.add((u, v))
                            edges.append((u, v))
                return n, edges
                
    # Fallback to LUW if no circulant found
    return _construct_luw(k, g)

def _check_girth_circulant(n, gens, g):
    # BFS to check if girth < g
    # We only need to check up to depth g//2 - 1
    max_depth = g // 2 - 1
    
    # dist[v] = distance from 0
    dist = [-1] * n
    dist[0] = 0
    queue = [(0, 0)]
    
    while queue:
        u, d = queue.pop(0)
        if d >= max_depth:
            continue
            
        for a in gens:
            # Neighbors are u+a and u-a
            for sign in [1, -1]:
                v = (u + sign * a) % n
                if dist[v] == -1:
                    dist[v] = d + 1
                    queue.append((v, d + 1))
                else:
                    # Found a cycle
                    # Cycle length is d + 1 + dist[v]
                    # Note: if v == u, it's a loop (length 1), but gens are non-zero.
                    # If v is parent, it's backtracking (length 2), but we check distinct dirs.
                    # The shortest cycle through this edge is d + 1 + dist[v]
                    # However, if dist[v] == d-1, it's the parent, cycle len 2.
                    # We want to detect cycles < g.
                    cycle_len = d + 1 + dist[v]
                    if cycle_len < g:
                        return False
    return True

def _construct_luw(k, g):
    # Original LUW construction
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
