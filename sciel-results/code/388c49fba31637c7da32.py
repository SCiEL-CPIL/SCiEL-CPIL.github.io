import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    time_limit = 110
    
    # Approach: Search for a circulant graph C_N(S) with girth >= g.
    # k=5, g=9.
    # N must be even.
    # S = { +/- s1, +/- s2, N/2 }
    
    def check_circulant_girth(N, s1, s2, k, g):
        # S = {s1, -s1, s2, -s2, N//2}
        # Note: -x mod N is N-x
        # If N is even, N//2 is its own inverse.
        # We need to ensure s1, s2 are distinct and not 0, N//2.
        
        if s1 == s2 or s1 == N//2 or s2 == N//2 or s1 == 0 or s2 == 0:
            return False
            
        S = [s1, N-s1, s2, N-s2, N//2]
        
        # BFS from 0
        # Target depth: (g-1)//2 = 4
        max_depth = (g - 1) // 2
        
        dist = [-1] * N
        dist[0] = 0
        q = deque([0])
        
        while q:
            u = q.popleft()
            d = dist[u]
            if d >= max_depth:
                continue
            for v_offset in S:
                v = (u + v_offset) % N
                if dist[v] == -1:
                    dist[v] = d + 1
                    q.append(v)
                else:
                    # Found a visited node.
                    # Cycle length is d + dist[v] + 1 (if v is not parent? No, in BFS tree, if v is visited, 
                    # the edge (u,v) creates a cycle of length d + dist[v] + 1 ONLY IF v is not the parent of u?
                    # Actually, if dist[v] <= d, the cycle length is d + dist[v] + 1?
                    # Let's check: Path 0->...->u (len d), edge u-v, path v->...->0 (len dist[v]).
                    # Total length d + 1 + dist[v].
                    # We need this to be >= g.
                    # Also, if v is the parent of u, dist[v] = d-1, cycle length d + 1 + d - 1 = 2d.
                    # We must ignore the parent edge to avoid detecting "cycles" of length 2 (which are just the edge itself).
                    # How to identify parent? We can store parent in dist or separate array.
                    # But for girth check, if we find ANY cycle < g, we return False.
                    # The "cycle" formed by back-edge to parent is length 2, which is < 9. 
                    # But back-edges to parent are NOT cycles in the graph sense, they are tree edges.
                    # In BFS, we only add edge to unvisited nodes. 
                    # So if dist[v] != -1, v is already visited.
                    # If v is the parent of u, we shouldn't count it as a cycle.
                    # But we don't store parent.
                    # However, if dist[v] == d-1 and v is the parent, the "cycle" is 2d? No, it's just the edge.
                    # Actually, if we encounter a visited node v from u:
                    # If dist[v] < d:
                    #   Cycle length = d + dist[v] + 1.
                    #   We need d + dist[v] + 1 >= g.
                    # If dist[v] == d:
                    #   Cycle length = 2d + 1? No, cross edge.
                    #   Cycle length is at least 2d+1?
                    #   Actually, if dist[v] == d, the shortest path from 0 to v is d, and from 0 to u is d.
                    #   The cycle is 0->...->u->v->...->0. Length d + 1 + d = 2d+1?
                    #   No, the path v->...->0 is length d.
                    #   So cycle length is d + 1 + d = 2d+1.
                    #   We need 2d+1 >= g.
                    # If dist[v] > d:
                    #   Symmetric to dist[v] < d.
                    
                    # Simplified check:
                    # The cycle length formed by edge (u,v) where dist[u]=d, dist[v]=d' is d + d' + 1.
                    # Exception: if v is the parent of u (d' = d-1), this is not a cycle.
                    # But we don't know if v is parent.
                    # However, if d' = d-1, the "cycle" length is 2d.
                    # If 2d < g, is it a valid cycle?
                    # In a tree, there are no cycles. The edge (u,v) is part of the tree.
                    # So we should ignore edges to parents.
                    # To do this correctly, we should store parent.
                    
                    pass # Logic below will use parent array
        
        # Re-do with parent tracking
        dist = [-1] * N
        parent = [-1] * N
        dist[0] = 0
        q = deque([0])
        
        while q:
            u = q.popleft()
            d = dist[u]
            if d >= max_depth:
                continue
            for v_offset in S:
                v = (u + v_offset) % N
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    q.append(v)
                else:
                    if v != parent[u] and parent[v] != u:
                        # It's a cross edge or back edge to non-parent
                        # Cycle length
                        if dist[v] < d:
                            cycle_len = d + dist[v] + 1
                        else:
                            cycle_len = d + dist[v] + 1 # Same formula works for undirected?
                            # Path 0->u (d), u-v (1), v->0 (dist[v]).
                            # Yes.
                        if cycle_len < g:
                            return False
        return True

    # Search for N
    # Start from Moore bound 428
    # Step by 2 (N must be even)
    
    # We will try random pairs for each N
    # Limit time
    
    for N in range(428, 1116, 2):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Number of trials per N
        # Smaller N: more trials?
        # Let's do 1000 trials for small N, 500 for large N?
        # Or just fixed 1000.
        trials = 1000
        if N > 800:
            trials = 500
            
        found = False
        for _ in range(trials):
            # Choose s1, s2 randomly from 1..N/2-1
            # Ensure s1 != s2
            s1 = random.randint(1, N//2 - 1)
            s2 = random.randint(1, N//2 - 1)
            if s1 == s2:
                continue
            if check_circulant_girth(N, s1, s2, k, g):
                # Construct edges
                edges = []
                # Vertices 0..N-1
                # Edges: (i, (i+s1)%N), (i, (i+s2)%N), (i, (i+N//2)%N)
                # To avoid duplicates, only add if i < j?
                # Or just add all and unique them?
                # For i in 0..N-1:
                #   j1 = (i+s1)%N
                #   if i < j1: add (i, j1)
                #   j2 = (i+s2)%N
                #   if i < j2: add (i, j2)
                #   j3 = (i+N//2)%N
                #   if i < j3: add (i, j3)
                
                for i in range(N):
                    j1 = (i + s1) % N
                    if i < j1:
                        edges.append((i, j1))
                    j2 = (i + s2) % N
                    if i < j2:
                        edges.append((i, j2))
                    j3 = (i + N//2) % N
                    if i < j3:
                        edges.append((i, j3))
                        
                return N, edges

    # Fallback: Algebraic construction from seed
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
