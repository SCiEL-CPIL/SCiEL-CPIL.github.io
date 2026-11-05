import random
import time
import math
import numpy as np
from collections import deque

def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Base graph: Petersen (10 vertices, 15 edges)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    n_edges = len(edges)
    adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Generate all closed non-backtracking walks of length 3 to g-1 from every vertex
    walks = []
    for start_v in range(n_base):
        queue = deque()
        for next_v, e_id in adj[start_v]:
            u, v = edges[e_id]
            sign = 1 if u == start_v else -1
            queue.append((next_v, e_id, 1, [(e_id, sign)]))
        
        while queue:
            curr_v, prev_e, length, path = queue.popleft()
            if length >= g:
                continue
            if curr_v == start_v and length >= 3:
                walks.append(path)
            for next_v, e_id in adj[curr_v]:
                if e_id == prev_e:
                    continue
                u, v = edges[e_id]
                sign = 1 if u == curr_v else -1
                new_path = path + [(e_id, sign)]
                queue.append((next_v, e_id, length + 1, new_path))
    
    unique_walks = set()
    for w in walks:
        unique_walks.add(tuple(w))
    walks = list(unique_walks)
    
    # Pre-compute coefficient matrix: rows = walks, cols = edges
    # Each entry is the signed count of edge usage in that walk
    walk_coeffs = np.zeros((len(walks), n_edges), dtype=np.int32)
    for wi, w in enumerate(walks):
        for e_id, sign in w:
            walk_coeffs[wi, e_id] += sign
    
    n_walks = len(walks)
    
    best_n = float('inf')
    best_edges = None
    best_m = None
    
    # Moore bound for n(3,16) is 512, so n = 10*m >= 512 means m >= 52
    for m in range(52, 2000):
        if time.time() > deadline - 8.0:
            break
        
        candidate_n = 10 * m
        time_left = deadline - time.time()
        
        # Estimate trials: vectorized check takes ~10-50us for 100 walks x 15 edges
        # Use adaptive budget
        max_tries = min(20000, max(500, int(time_left / 0.00005)))
        
        found = False
        for trial in range(max_tries):
            if time.time() > deadline - 2.0:
                break
            
            # Strategy: random voltages in [1, m-1]
            # Use numpy for speed
            volts = np.random.randint(1, m, size=n_edges).astype(np.int64)
            
            # Quick connectivity check
            g_val = m
            for v in volts:
                g_val = math.gcd(g_val, int(v))
                if g_val == 1:
                    break
            if g_val != 1:
                continue
            
            # Check girth: no walk should have voltage sum ≡ 0 (mod m)
            # Compute all walk sums vectorized
            sums = walk_coeffs @ volts  # shape (n_walks,)
            if not np.any(sums % m == 0):
                found = True
                break
        
        if found:
            best_m = m
            best_n = candidate_n
            lift_edges = []
            for i, (u, v) in enumerate(edges):
                vol = int(volts[i])
                for t in range(m):
                    u_lift = u * m + t
                    v_lift = v * m + (t + vol) % m
                    lift_edges.append((u_lift, v_lift))
            best_edges = lift_edges
            break
    
    if best_edges is None:
        return _luw_fallback(k, g)
    
    return best_n, best_edges

def _luw_fallback(k, g):
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    return len(order), edges_luw

if __name__ == "__main__":
    n, e = construct(3, 16)
    print(f"n={n}")
