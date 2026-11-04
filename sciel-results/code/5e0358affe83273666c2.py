import math
import time
from collections import deque

def seed_construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
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

def check_girth_gp(n, k_val, g):
    # BFS from vertex 0 (u_0)
    # Vertices: 0..n-1 are u_i, n..2n-1 are v_i
    dist = [-1] * (2 * n)
    parent = [-1] * (2 * n)
    dist[0] = 0
    q = deque([0])
    
    while q:
        u = q.popleft()
        du = dist[u]
        
        # If we have gone deep enough, any further cycles will be long enough
        if du >= g // 2:
            continue
            
        if u < n:
            i = u
            # Neighbors: u_{i-1}, u_{i+1}, v_i
            neighs = [
                (i - 1) % n,
                (i + 1) % n,
                n + i
            ]
        else:
            i = u - n
            # Neighbors: u_i, v_{i-k}, v_{i+k}
            neighs = [
                i,
                n + (i - k_val) % n,
                n + (i + k_val) % n
            ]
            
        for v in neighs:
            if dist[v] == -1:
                dist[v] = du + 1
                parent[v] = u
                q.append(v)
            else:
                if v != parent[u]:
                    cycle_len = du + 1 + dist[v]
                    if cycle_len < g:
                        return False
    return True

def construct(k, g):
    if k != 3:
        return seed_construct(k, g)
        
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Moore bound for k=3
    d = (g - 1) // 2
    if d == 0:
        moore = 2
    else:
        # 1 + 3 * (2^(d+1) - 1)
        moore = 1 + 3 * ((1 << (d + 1)) - 1)
        
    n_min = (moore + 1) // 2
    
    best_n = None
    best_k = None
    
    # Search for Generalized Petersen Graph GP(n, k)
    # We check n from n_min upwards
    # Limit the search range to avoid infinite loops if no solution exists in this family
    # 500 iterations of n is usually enough for small girths
    for n in range(n_min, n_min + 1000):
        if time.time() > deadline:
            break
            
        # k_val must be coprime to n and 1 <= k_val < n/2
        # To ensure simple graph and avoid duplicates in edge list logic
        for k_val in range(1, n // 2 + 1):
            if time.time() > deadline:
                break
                
            if math.gcd(n, k_val) != 1:
                continue
                
            if check_girth_gp(n, k_val, g):
                best_n = n
                best_k = k_val
                break
                
        if best_n is not None:
            break
            
    if best_n is None:
        # Fallback to seed
        return seed_construct(k, g)
        
    # Construct the graph
    n = best_n
    k_val = best_k
    edges = []
    
    for i in range(n):
        # u_i to u_{i+1}
        edges.append((i, (i + 1) % n))
        # u_i to v_i
        edges.append((i, n + i))
        
    for i in range(n):
        # v_i to v_{i+k}
        j = (i + k_val) % n
        if i < j:
            edges.append((n + i, n + j))
            
    return 2 * n, edges
