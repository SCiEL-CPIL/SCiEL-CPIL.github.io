import numpy as np
import random
import time
from collections import deque

def construct(k, g):
    deadline = time.time() + 115

    # For the cage problem n(4, 10), the smallest known is 320 vertices.
    # We attempt to find a 4-regular graph on n vertices with girth >= g.
    
    # Moore bound for k=4, g=10:
    # g even: n >= k * sum_{i=0}^{g/2-1} (k-1)^i = 4 * (1 + 3 + 9 + 27 + 81) = 4 * 121 = 484
    # Wait, that's not right. Moore bound for girth g (even) is:
    # n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2 * (1 + 3 + 9 + 27 + 81) = 2 * 121 = 242
    # So lower bound is 242, and known minimum is 320.
    
    # Strategy: Use known cage constructions. For n(4,10), the smallest known has 320 vertices.
    # We'll use a voltage construction over a smaller base graph.
    
    # The (3,6) cage (Tutte-Coxeter graph) has 30 vertices, girth 8, 3-regular.
    # The (4,8) cage has 144 vertices, girth 8, 4-regular.
    # For girth 10, we need to be more careful.
    
    # Let's try a direct construction using the known 320-vertex cage if we can find it,
    # or use a random regular graph approach with girth checking.
    
    # Actually, let's use a known parametrized family. The "circulant" or "voltage" approach.
    
    # For k=4, g=10, we know the answer is 320. Let's try to construct it.
    # One known construction uses the incidence graph of a symmetric design or a specific
    # voltage lift.
    
    # Let's try the following: Use a base graph B with n_b vertices, and lift by a group G
    # of order m to get n = n_b * m vertices. We need girth >= 10 in the lift.
    
    # A good base is the (4,6) cage or similar. But let's use a different approach.
    
    # Actually, for this specific problem, let's use a known construction.
    # The 320-vertex (4,10)-cage can be constructed as follows:
    # It is the point-line graph of a generalized quadrangle or similar.
    
    # Let's try a simpler approach: random 4-regular graphs and check girth.
    # But finding a 4-regular graph on 320 vertices with girth >= 10 randomly is hard.
    
    # Let's use the "Higman-Sims" related construction or the "Sylvester" construction.
    
    # Actually, let's just implement a known good construction for n(4,10).
    # The graph can be described as follows:
    # Vertices: (i, j) where i in Z_20, j in Z_16, so 320 vertices.
    # But this is complex. Let me try a different known construction.
    
    # For now, let's use a voltage construction over Z_320 or a product structure.
    
    # A known (4,10)-cage: It can be constructed from a (4,10)-BPSG.
    
    # Let's try to use the "circulant graph" approach with a clever connection set.
    
    # Actually, the simplest reliable approach for this specific task is to use the
    # known 320-vertex construction. Let me code up the one from the literature.
    
    # The (4,10) cage: vertices are elements of Z_20 x Z_16 = Z_320 (cyclic interpretation).
    # Connection: each vertex v is connected to v ± a, v ± b for some a, b.
    # But 4-regular circulant on 320 vertices: C_320(±a, ±b).
    
    # Let me search for the right a, b such that girth >= 10.
    
    # For a circulant graph C_n(±a, ±b), the girth is determined by the shortest
    # relations: a_i * a ± a_j * b = 0 mod n with minimal |a_i| + |a_j|.
    
    # We need to find a, b in Z_320 such that:
    # 1. gcd(a, 320) = gcd(b, 320) = 1 (to ensure 4-regular, no multiple edges)
    # 2. The shortest non-trivial relation i*a + j*b = 0 mod 320 with i,j integers
    #    has |i| + |j| >= 5 (girth = 2*(|i|+|j|) >= 10 means |i|+|j| >= 5)
    #    Wait, girth 10 means shortest cycle length 10, so |i|+|j| = 5 for the shortest.
    
    # Let's try to find such a, b.
    
    n_target = 320
    
    best_a, best_b = None, None
    for a in range(1, n_target // 2 + 1):
        if time.time() > deadline - 5:
            break
        if __import__('math').gcd(a, n_target) != 1:
            continue
        for b in range(1, n_target // 2 + 1):
            if __import__('math').gcd(b, n_target) != 1:
                continue
            if a == b:
                continue
            
            # Find shortest cycle: min |i| + |j| such that i*a + j*b = 0 mod n_target
            # with i, j having the same "direction" signs... actually for undirected:
            # We need i*a ≡ j*b mod n_target, i.e., i*a - j*b = 0 mod n_target
            # or i*a + j*b = 0 mod n_target, etc.
            
            # The cycle length is |i| + |j| where i*a + j*b = 0 mod n_target,
            # but we need to be careful about the signs.
            
            # Actually for C_n(±a, ±b), a cycle of length 2m exists iff
            # there are integers i_1, ..., i_m with i_k in {±1, ±2} (coefficient of a or b)
            # such that sum = 0 mod n.
            
            # Simpler: find min |i| + |j| where i*a + j*b ≡ 0 (mod n), i,j >= 0, not both 0.
            # But this isn't quite right either.
            
            # Let's just build the graph and check girth for a few candidates.
            
            if time.time() > deadline - 10:
                break
            
            # Quick check: build graph and compute girth
            adj = [[] for _ in range(n_target)]
            for v in range(n_target):
                adj[v] = [(v + a) % n_target, (v - a) % n_target, 
                          (v + b) % n_target, (v - b) % n_target]
            
            # Check simple: no loops (a != 0, b != 0 mod n, and 2a != 0, 2b != 0 mod n)
            if 2 * a % n_target == 0 or 2 * b % n_target == 0:
                continue
            # Check no repeated edges: a != b, a != -b, etc.
            if a == b or a == (n_target - b) or (n_target - a) == b:
                continue
            
            # BFS for girth
            girth = n_target + 1
            for start in range(n_target):
                dist = [-1] * n_target
                parent = [-1] * n_target
                dist[start] = 0
                queue = deque([start])
                while queue:
                    cur = queue.popleft()
                    for nb in adj[cur]:
                        if dist[nb] == -1:
                            dist[nb] = dist[cur] + 1
                            parent[nb] = cur
                            queue.append(nb)
                        elif parent[nb] != cur and parent[cur] != nb:
                            cl = dist[nb] + dist[cur] + 1
                            if cl < girth:
                                girth = cl
                if girth < g:
                    break
            
            if girth >= g:
                best_a, best_b = a, b
                break
        if best_a is not None:
            break
    
    if best_a is not None:
        edges = []
        for v in range(n_target):
            for nb in [(v + best_a) % n_target, (v - best_a) % n_target,
                        (v + best_b) % n_target, (v - best_b) % n_target]:
                if v < nb:
                    edges.append((v, nb))
        return n_target, edges
    
    # Fallback: LUW-type construction
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
