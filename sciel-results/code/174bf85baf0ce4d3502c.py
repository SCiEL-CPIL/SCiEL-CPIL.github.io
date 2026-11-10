import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 115.0  # Leave 5s margin for 120s limit

    # Fallback: L-U-W construction
    def luw_construct(k, g):
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

    # Moore bound
    if g % 2 == 0:
        moore_bound = 1 + k * sum((k-1)**i for i in range(g//2 - 1))
    else:
        moore_bound = 2 * sum((k-1)**i for i in range((g-1)//2))
        
    search_start = max(moore_bound, 243) if (k == 4 and g == 10) else moore_bound

    def check_girth(n, generators):
        # Optimized BFS using deque and bitsets if possible, but set is fine for small n
        visited = set()
        queue = deque()
        queue.append((0, -1, 0))
        visited.add((0, -1))
        
        while queue:
            pos, last_idx, depth = queue.popleft()
            
            if depth >= g:
                return True
            
            if pos == 0 and depth > 0:
                return depth >= g
            
            for i, gen in enumerate(generators):
                next_pos = (pos + gen) % n
                if last_idx != -1:
                    if gen == (-generators[last_idx]) % n:
                        continue
                
                state = (next_pos, i)
                if state not in visited:
                    visited.add(state)
                    queue.append((next_pos, i, depth + 1))
        
        return True

    def is_valid_generators(n, generators):
        if any(gen == 0 for gen in generators):
            return False
        gen_set = set(generators)
        if len(gen_set) != k:
            return False
        for gen in generators:
            if (-gen) % n in gen_set:
                return False
        import math
        g_val = n
        for gen in generators:
            g_val = math.gcd(g_val, gen)
        if g_val != 1:
            return False
        return check_girth(n, generators)

    current_n = search_start
    max_n = 1000
    best_n = 512 # Default L-U-W size
    best_edges = None

    # Try to find a better graph
    while current_n <= max_n and time.time() < deadline:
        attempts = 0
        max_attempts = 2000 if current_n < 300 else 5000
        found = False
        
        while attempts < max_attempts and time.time() < deadline:
            used = set()
            gens = []
            valid = True
            for _ in range(k):
                gen = random.randint(1, n - 1)
                if gen in used or (-gen) % n in used:
                    valid = False
                    break
                used.add(gen)
                gens.append(gen)
            
            if not valid:
                attempts += 1
                continue
                
            if is_valid_generators(current_n, gens):
                edges = []
                for u in range(current_n):
                    for v_gen in gens:
                        v = (u + v_gen) % current_n
                        if u < v:
                            edges.append((u, v))
                return current_n, edges
        
        current_n += 1

    # Fallback to L-U-W
    return luw_construct(k, g)
