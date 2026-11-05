import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = 110.0
    
    # Known minimum for n(4,10) is 320.
    # Moore bound is 316.
    # We try to find a graph of size 320.
    # If that fails, we try slightly larger sizes, or fallback.
    
    target_sizes = [320, 324, 328, 332, 336, 340, 344, 348, 352, 356, 360]
    
    for n in target_sizes:
        if time.time() > deadline:
            break
            
        # Try to construct
        # Increase attempts if we have more time?
        # For n=320, try 2000 attempts.
        attempts = 2000
        for i in range(attempts):
            if time.time() > deadline:
                break
            res = try_construct_greedy(n, k, g)
            if res is not None:
                return res
                
    # Fallback: LUW construction
    return luw_construct(k, g)

def try_construct_greedy(n, k, g):
    # k-regular graph on n vertices
    # Greedy random edge addition with girth check
    
    adj = [[] for _ in range(n)]
    deg = [0] * n
    edges = []
    
    # Adjacency matrix for fast edge check
    # Using list of bytearrays
    mat = [bytearray(n) for _ in range(n)]
    
    max_dist = (g - 2) // 2
    
    # Pre-allocate dist arrays
    dist_u = [-1] * n
    dist_v = [-1] * n
    
    for _ in range(k * n // 2):
        # Find vertex with max remaining degree
        max_rem = -1
        u = -1
        for i in range(n):
            rem = k - deg[i]
            if rem > max_rem:
                max_rem = rem
                u = i
                
        if max_rem == 0:
            return (n, edges)
            
        # Find valid v
        # Shuffle candidates?
        # Candidates: i != u, deg[i] < k, not connected
        candidates = []
        for i in range(n):
            if i != u and deg[i] < k and mat[u][i] == 0:
                candidates.append(i)
                
        random.shuffle(candidates)
        
        found = False
        for v in candidates:
            # Check girth
            if is_valid_edge(u, v, adj, mat, max_dist, g, dist_u, dist_v):
                # Add edge
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1
                mat[u][v] = 1
                mat[v][u] = 1
                edges.append((u, v))
                found = True
                break
                
        if not found:
            return None # Stuck, restart
            
    return (n, edges)

def is_valid_edge(u, v, adj, mat, max_dist, g, dist_u, dist_v):
    # Reset dist arrays
    # Instead of resetting whole array, use a timestamp or just reset the visited nodes
    # For simplicity and n=320, resetting is fast enough?
    # 320 assignments * 2 * 160 edges * 2000 attempts = 204 million assignments.
    # Might be slow.
    # Let's use a "visited" list to only reset what we touched.
    
    touched_u = []
    touched_v = []
    
    dist_u[u] = 0
    touched_u.append(u)
    dist_v[v] = 0
    touched_v.append(v)
    
    q_u = [u]
    q_v = [v]
    
    # BFS u
    for _ in range(max_dist):
        next_q_u = []
        for node in q_u:
            d = dist_u[node] + 1
            for nei in adj[node]:
                if dist_u[nei] == -1:
                    dist_u[nei] = d
                    touched_u.append(nei)
                    next_q_u.append(nei)
        q_u = next_q_u
        
    # BFS v
    for _ in range(max_dist):
        next_q_v = []
        for node in q_v:
            d = dist_v[node] + 1
            for nei in adj[node]:
                if dist_v[nei] == -1:
                    dist_v[nei] = d
                    touched_v.append(nei)
                    next_q_v.append(nei)
        q_v = next_q_v
        
    # Check intersection
    # We can check during BFS?
    # If we find a node in q_u that is already visited by v (and dist sum < g), return False.
    # But we do BFS fully first.
    
    for node in touched_u:
        if dist_v[node] != -1:
            if dist_u[node] + dist_v[node] + 1 < g:
                # Cleanup
                for x in touched_u: dist_u[x] = -1
                for x in touched_v: dist_v[x] = -1
                return False
                
    # Cleanup
    for x in touched_u: dist_u[x] = -1
    for x in touched_v: dist_v[x] = -1
    return True

def luw_construct(k, g):
    # Fallback from seed
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
