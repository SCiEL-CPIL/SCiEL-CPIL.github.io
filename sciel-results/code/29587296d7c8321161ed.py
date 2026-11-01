import time
import random

def construct(k, g):
    # Fallback seed logic
    def seed_construct():
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

    if k != 3 or g != 16:
        return seed_construct()

    t0 = time.time()
    deadline = t0 + 140
    
    # Petersen graph edges
    edges_petersen = [(0,1), (1,2), (2,3), (3,4), (4,0),
                      (5,6), (6,7), (7,8), (8,9), (9,5),
                      (0,5), (1,6), (2,7), (3,8), (4,9)]
    
    # Precompute cycles
    adj = {i: [] for i in range(10)}
    for u, v in edges_petersen:
        adj[u].append(v)
        adj[v].append(u)
        
    cycles = []
    def dfs(current, visited, path):
        if len(path) > 15:
            return
        if len(path) >= 3 and path[0] == current:
            cycles.append(path[:])
            return
        for neighbor in adj[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, visited, path)
                path.pop()
                visited.remove(neighbor)
                
    for start in range(10):
        visited = {start}
        path = [start]
        dfs(start, visited, path)
        
    unique_cycles = []
    seen = set()
    for c in cycles:
        min_idx = c.index(min(c))
        c_rot = c[min_idx:] + c[:min_idx]
        c_rev = list(reversed(c_rot))
        if tuple(c_rot) < tuple(c_rev):
            key = tuple(c_rot)
        else:
            key = tuple(c_rev)
        if key not in seen:
            seen.add(key)
            unique_cycles.append(key)
            
    best_n = float('inf')
    best_voltages = None
    best_m = None
    
    for m in range(77, 150):
        if time.time() > deadline:
            break
            
        for attempt in range(500):
            if time.time() > deadline:
                break
                
            volt = {}
            for u, v in edges_petersen:
                vol = random.randint(0, m-1)
                volt[(u, v)] = vol
                volt[(v, u)] = (-vol) % m
                
            valid = True
            for cycle in unique_cycles:
                s = 0
                for i in range(len(cycle)-1):
                    u = cycle[i]
                    v = cycle[i+1]
                    s = (s + volt[(u, v)]) % m
                u = cycle[-1]
                v = cycle[0]
                s = (s + volt[(u, v)]) % m
                
                if s == 0:
                    valid = False
                    break
                    
            if valid:
                n = 10 * m
                if n < best_n:
                    best_n = n
                    best_m = m
                    best_voltages = volt
                if n <= 770:
                    break
        if best_n <= 770:
            break
            
    if best_n == float('inf'):
        return seed_construct()

    edges_out = []
    for u, v in edges_petersen:
        vol = best_voltages[(u, v)]
        for i in range(best_m):
            id_u = u * best_m + i
            id_v = v * best_m + (i + vol) % best_m
            edges_out.append((id_u, id_v))
            
    return best_n, edges_out
