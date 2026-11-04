import random
import time

def construct(k, g):
    # If the specific target is not (3, 16), fall back to the original LUW construction
    if k != 3 or g != 16:
        return _construct_luw(k, g)
    
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Try voltage graph construction on K4
    result = _construct_voltage_k4(3, 16, time_limit, start_time)
    if result:
        return result
        
    # Fallback to LUW if no voltage graph found
    return _construct_luw(k, g)

def _construct_voltage_k4(k, g, time_limit, start_time):
    # Base graph K4: vertices 0,1,2,3
    # We fix v01=1, v02=1, v03=1, v12=1
    # Search for v13, v23 in Z_m
    
    # Generate non-backtracking closed walks in K4 of length < g
    max_len = g - 1
    walks = []
    
    def gen_walks(current, prev, path, length):
        if length > max_len:
            return
        if length > 0 and current == 0:
            walks.append(list(path))
        
        for next_v in range(4):
            if next_v == prev:
                continue
            if next_v == current:
                continue
            path.append(next_v)
            gen_walks(next_v, current, path, length + 1)
            path.pop()
            
    gen_walks(0, -1, [0], 0)
    
    # Compute signatures (const, c13, c23) assuming v12=1
    # Edges: (0,1)->1, (0,2)->1, (0,3)->1, (1,2)->1
    # (1,3)->v13, (2,3)->v23
    walk_sigs = []
    for w in walks:
        const = 0
        c13 = 0
        c23 = 0
        for i in range(len(w) - 1):
            u, v = w[i], w[i+1]
            if (u == 0 and v == 1) or (u == 1 and v == 0):
                const += 1 if u == 0 else -1
            elif (u == 0 and v == 2) or (u == 2 and v == 0):
                const += 1 if u == 0 else -1
            elif (u == 0 and v == 3) or (u == 3 and v == 0):
                const += 1 if u == 0 else -1
            elif (u == 1 and v == 2) or (u == 2 and v == 1):
                const += 1 if u == 1 else -1
            elif (u == 1 and v == 3) or (u == 3 and v == 1):
                c13 += 1 if u == 1 else -1
            elif (u == 2 and v == 3) or (u == 3 and v == 2):
                c23 += 1 if u == 2 else -1
        walk_sigs.append((const, c13, c23))
        
    # Sort by length (approximated by index in generation, but we can just check all)
    # Actually, shorter walks are more likely to fail, so we check them first.
    # My generation is BFS-like? No, DFS.
    # Let's sort by length of walk
    walk_sigs_with_len = [(len(w), s) for w, s in zip(walks, walk_sigs)]
    walk_sigs_with_len.sort(key=lambda x: x[0])
    
    # Search for m
    for m in range(128, 250):
        if time.time() - start_time > time_limit:
            return None
            
        for _ in range(5000):
            if time.time() - start_time > time_limit:
                return None
                
            v13 = random.randint(0, m-1)
            v23 = random.randint(0, m-1)
            
            valid = True
            for length, (const, c13, c23) in walk_sigs_with_len:
                s = const + c13*v13 + c23*v23
                if s % m == 0:
                    valid = False
                    break
            
            if valid:
                # Construct graph
                n = 4 * m
                edges = []
                seen = set()
                base_edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
                voltages = {
                    (0,1): 1, (1,0): -1,
                    (0,2): 1, (2,0): -1,
                    (0,3): 1, (3,0): -1,
                    (1,2): 1, (2,1): -1,
                    (1,3): v13, (3,1): -v13,
                    (2,3): v23, (3,2): -v23
                }
                
                for u_base, v_base in base_edges:
                    vol = voltages[(u_base, v_base)]
                    for i in range(m):
                        u = u_base * m + i
                        v_idx = (i + vol) % m
                        v = v_base * m + v_idx
                        u, v = min(u, v), max(u, v)
                        if (u, v) not in seen:
                            seen.add((u, v))
                            edges.append((u, v))
                
                return n, edges
    return None

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
