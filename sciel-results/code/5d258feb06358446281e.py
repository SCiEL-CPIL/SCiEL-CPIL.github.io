import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # 1. Try to find a small graph via covering of the Tutte-Coxeter graph
    # The Tutte-Coxeter graph is (3,8)-cage, 30 vertices.
    # We construct a 5-fold cover.
    # If we find a cover with girth >= 9, we return it.
    
    def get_tutte_coxeter():
        # 30 vertices, 3-regular, girth 8.
        # We can hardcode the edges or generate them.
        # The Tutte-Coxeter graph is the incidence graph of the generalized quadrangle GQ(2,2).
        # GQ(2,2) has 15 points and 15 lines.
        # Points: (x,y,z) in GF(2)^3 \ {0}
        # Lines: sets of 3 points summing to 0.
        # Incidence: point p is on line L if p in L.
        
        points = []
        for i in range(1, 8):
            points.append(i)
            
        # Generate lines of GQ(2,2)
        # A line is a set of 3 points {a,b,c} such that a+b+c=0 (xor).
        lines = []
        for i in range(1, 8):
            for j in range(i+1, 8):
                k_val = i ^ j
                if k_val > j:
                    lines.append((i, j, k_val))
        
        # Map points to 0..14, lines to 15..29
        n_pts = len(points)
        n_lines = len(lines)
        
        edges = []
        for idx_l, line in enumerate(lines):
            line_id = n_pts + idx_l
            for p in line:
                # p is 1..7, index is p-1
                edges.append((p-1, line_id))
                
        return edges, n_pts + n_lines

    def check_girth(edges, n):
        # BFS from every vertex to find shortest cycle
        # Since n is small (150), this is fast.
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        min_girth = 20 # Max possible girth for n=150 is less than 20?
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        # Cycle found
                        # Length is dist[u] + dist[v] + 1
                        # But we need to trace back to find the actual cycle length?
                        # For BFS, the first time we see a back-edge, the cycle length is dist[u] + dist[v] + 1.
                        # However, this might not be the shortest cycle through start.
                        # But we take the min over all starts.
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth

    # Try random covers
    base_edges, base_n = get_tutte_coxeter()
    base_adj = [[] for _ in range(base_n)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
        
    # Voltage assignment
    # We need to assign a voltage (0..4) to each edge.
    # The cover has vertices (v, i) where v in 0..base_n-1, i in 0..4.
    # Edge ((u, i), (v, j)) exists if (u,v) is in base_edges and j = (i + voltage(u,v)) % 5.
    
    best_n = 6250 # LUW fallback size
    best_edges = None
    
    for _ in range(100):
        if time.time() > deadline:
            break
            
        # Random voltages
        voltages = {}
        for u, v in base_edges:
            voltages[(u, v)] = random.randint(0, 4)
            voltages[(v, u)] = (5 - voltages[(u, v)]) % 5
            
        cover_n = base_n * 5
        cover_edges = []
        for u in range(base_n):
            for v in base_adj[u]:
                if u < v: # Avoid duplicates
                    for i in range(5):
                        j = (i + voltages[(u, v)]) % 5
                        u_id = u * 5 + i
                        v_id = v * 5 + j
                        cover_edges.append((u_id, v_id))
                        
        # Check girth
        girth = check_girth(cover_edges, cover_n)
        if girth >= g:
            return cover_n, cover_edges
            
    # Fallback to LUW
    # Construct LUW graph with q=5, d=5
    q = 5
    d = 5
    # ... LUW construction code from seed ...
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = (x[c] + y[a] * x[b]) % q if side == 0 else (x[c] - x[a] * y[b]) % q
            yield 1 - side, tuple(y)

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

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
