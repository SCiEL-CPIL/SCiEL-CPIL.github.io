import random
import time

def construct(k, g):
    if k != 3 or g != 16:
        return _construct_luw(k, g)
    
    start_time = time.time()
    time_limit = 110
    
    # Target the known 936-vertex (3,16)-cage using voltage graph construction
    n_target = 936
    N = n_target // 2  # 468
    
    # Try to find voltages v1, v2, v3 in Z_N such that girth >= 16
    # This means no non-backtracking closed walk of length < 16 has sum 0.
    
    def check_voltage_girth(v1, v2, v3, N, min_girth):
        # Non-backtracking walks in K_2^3
        # Edges 0, 1, 2 correspond to v1, v2, v3
        # Walks start at 0, alternate 0-1-0...
        # Length L must be even.
        # Sequence of edge indices e1, e2, ..., eL such that ej != e(j-1)
        # Sum = v[e1] - v[e2] + v[e3] ...
        
        max_len = min_girth - 1 # We check L < min_girth
        # Actually, girth is the length of the shortest cycle.
        # If a walk of length L < min_girth sums to 0, girth <= L.
        # So we check L = 2, 4, ..., min_girth - 1 (if min_girth is even, L < min_girth).
        
        # Generate walks
        # For efficiency, we can precompute or just iterate.
        # Number of walks is small.
        
        for L in range(2, min_girth, 2):
            # Recursive generation or iterative
            # We can use a stack or recursion.
            # Given the small count, recursion is fine.
            
            def gen_walks(prev_edge, depth, current_sum):
                if depth == L:
                    if current_sum % N == 0:
                        return True # Found short cycle
                    return False
                
                for e in range(3):
                    if e == prev_edge:
                        continue
                    sign = 1 if (depth % 2 == 0) else -1 # 0-indexed depth?
                    # Let's trace:
                    # Depth 0 (first edge): sign +
                    # Depth 1 (second edge): sign -
                    # So sign = 1 if depth is even, -1 if odd.
                    new_sum = current_sum + sign * [v1, v2, v3][e]
                    if gen_walks(e, depth + 1, new_sum):
                        return True
                return False
            
            # Start with any edge
            for e0 in range(3):
                if gen_walks(e0, 1, [v1, v2, v3][e0]):
                    return False # Invalid voltages
        
        return True

    # Search for valid voltages
    for _ in range(10000):
        if time.time() - start_time > time_limit:
            break
            
        v1 = random.randint(1, N - 1)
        v2 = random.randint(1, N - 1)
        v3 = random.randint(1, N - 1)
        
        if v1 == v2 or v2 == v3 or v1 == v3:
            continue
            
        if check_voltage_girth(v1, v2, v3, N, 16):
            # Construct graph
            edges = []
            seen = set()
            # Vertices 0..N-1 are side 0, N..2N-1 are side 1?
            # Or 0..N-1 are (0, x), N..2N-1 are (1, x).
            # Let's map (side, x) to index.
            # index = side * N + x
            
            for x in range(N):
                # Edges from (0, x) to (1, x+v1), (1, x+v2), (1, x+v3)
                for v in [v1, v2, v3]:
                    y = (x + v) % N
                    u = x # side 0
                    w = N + y # side 1
                    if u < w:
                        pair = (u, w)
                    else:
                        pair = (w, u)
                    if pair not in seen:
                        seen.add(pair)
                        edges.append(pair)
                        
            return n_target, edges
            
    # Fallback
    return _construct_luw(k, g)

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
