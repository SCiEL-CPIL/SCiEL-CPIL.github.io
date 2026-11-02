import time
import sys
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    # Precompute non-backtracking walk coefficient pairs (c1, c2) for lengths 3..g-1
    # c1 = net number of steps of type 1 (a) minus type -1
    # c2 = net number of steps of type 2 (b) minus type -2
    valid_coeffs = set()
    if g > 3:
        states = set()
        # Length 1
        states.add((1, 1, 0))
        states.add((-1, -1, 0))
        states.add((2, 0, 1))
        states.add((-2, 0, -1))
        
        for L in range(2, g):
            new_states = set()
            for last_step, c1, c2 in states:
                for step in [1, -1, 2, -2]:
                    if step == -last_step:
                        continue
                    nc1 = c1 + (1 if step == 1 else -1 if step == -1 else 0)
                    nc2 = c2 + (1 if step == 2 else -1 if step == -2 else 0)
                    new_states.add((step, nc1, nc2))
            states = new_states
            if L >= 3:
                for _, c1, c2 in states:
                    valid_coeffs.add((c1, c2))

    # Moore bound for even girth g=2R
    R = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(R))

    best_n = None
    best_edges = None

    # Search for circulant graph
    # n must be at least moore_bound
    # We search n in [moore_bound, moore_bound + 300]
    # For each n, we try random offsets

    best_n = None
    best_edges = None

    # Try deterministic small offsets first?
    # Or just random.
    # Let's try a mix.

    n_max_search = moore_bound + 300

    for n in range(moore_bound, n_max_search + 1):
        if time.time() > deadline - 5:
            break

        # Try random offsets
        # Number of trials per n
        trials = 150
        found = False

        # Shuffle offsets?
        # We can just generate random a, b

        for _ in range(trials):
            # a, b in [1, n//2 - 1]
            # Ensure a != b
            a = random.randint(1, n // 2 - 1)
            b = random.randint(1, n // 2 - 1)
            if a == b:
                continue

            # Check girth
            valid = True
            for c1, c2 in valid_coeffs:
                if (c1 * a + c2 * b) % n == 0:
                    valid = False
                    break

            if valid:
                # Construct edges
                edges = set()
                for i in range(n):
                    j1 = (i + a) % n
                    if i < j1:
                        edges.add((i, j1))
                    else:
                        edges.add((j1, i))

                    j2 = (i + b) % n
                    if i < j2:
                        edges.add((i, j2))
                    else:
                        edges.add((j2, i))

                best_n = n
                best_edges = list(edges)
                found = True
                break

        if found:
            return best_n, best_edges

    # Fallback to LUW
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return x * y % q

    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    queue = [0]
    head = 0
    
    while head < len(order):
        if time.time() > deadline:
            break
        v_idx = queue[head]
        head += 1
        v = order[v_idx]
        for other in neighbours(v):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)
            if v[0] == 0:
                edges.append((v_idx, index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
