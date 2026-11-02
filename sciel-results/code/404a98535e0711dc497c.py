import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # For k=4, g=10: try voltage graph approach on K_{4,4}
    # The lifted graph has 8*m vertices
    # We need to find voltages such that no cycle shorter than g exists
    
    best_result = None
    best_n = float('inf')
    
    # Try multiple m values, starting from known good range
    # Known best is 320 (m=40), lower bound is 243
    # Try m values that give n = 8*m
    for m in range(31, 55):
        if time.time() > deadline - 10:
            break
        n_vertices = 8 * m
        
        # Multiple random restarts
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize random voltages
            voltages = {}
            for a in range(4):
                for b in range(4):
                    voltages[(a, b)] = random.randint(0, m - 1)
            
            # Local search with efficient girth checking
            edges = build_edges_fast(m, voltages)
            min_g = check_girth_fast(n_vertices, edges, g)
            
            if min_g >= g:
                if n_vertices < best_n:
                    best_n = n_vertices
                    best_result = (n_vertices, edges)
                break
            
            # Local search: try to improve girth
            max_iterations = 200
            for it in range(max_iterations):
                if time.time() > deadline - 3:
                    break
                
                # Pick a random voltage to change
                a = random.randint(0, 3)
                b = random.randint(0, 3)
                old_v = voltages[(a, b)]
                new_v = random.randint(0, m - 1)
                if new_v == old_v:
                    continue
                
                voltages[(a, b)] = new_v
                edges = build_edges_fast(m, voltages)
                new_min_g = check_girth_fast(n_vertices, edges, g)
                
                if new_min_g >= g:
                    if n_vertices < best_n:
                        best_n = n_vertices
                        best_result = (n_vertices, edges)
                    break
                elif new_min_g > min_g:
                    min_g = new_min_g
                else:
                    # Revert
                    voltages[(a, b)] = old_v
            
            if best_result is not None:
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: LUW construction
    return luw_construct(k, g)

def build_edges_fast(m, voltages):
    """Build edges for voltage graph on K_{4,4}"""
    edges = []
    for a in range(4):
        for b in range(4):
            v = voltages[(a, b)]
            base_a = a
            base_b = 4 + b
            for i in range(m):
                u_idx = base_a * m + i
                j_idx = (i + v) % m
                v_idx = base_b * m + j_idx
                edges.append((u_idx, v_idx))
    return edges

def check_girth_fast(n, edges, g):
    """
    Fast girth check: returns the minimum cycle length, or g if no cycle < g found.
    Uses BFS from each vertex up to depth g//2.
    """
    if n == 0:
        return g
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_g = float('inf')
    max_depth = (g - 1) // 2  # For g=10, max_depth=4
    
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            if dist[u] >= max_depth:
                continue
            
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif dist[v] <= dist[u]:
                    # Back edge or cross edge
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_g:
                        min_g = cycle_len
                        if min_g < g:
                            return min_g
    
    return min_g if min_g < float('inf') else g

def luw_construct(k, g):
    """LUW/LW graph construction as fallback"""
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
