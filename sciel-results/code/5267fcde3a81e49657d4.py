import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Voltage graph construction over Petersen graph
    # Petersen graph: 10 vertices, 15 edges, girth 5, 3-regular
    # A regular cover of degree d has 10*d vertices
    # We assign voltages from Z_d to edges so the lift has girth >= g
    
    # Petersen graph edges (standard labeling)
    # Outer pentagon: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
    ]
    
    # For each edge, we need a voltage in Z_d
    # The lift has vertices (v, s) where v is a Petersen vertex and s is in Z_d
    # Edge (u,v) with voltage a: (u,s) connects to (v, s+a)
    # A closed walk in the base with voltage sum = 0 mod d lifts to a cycle
    
    # Strategy: try random voltage assignments and check girth of the lift
    # For d=96, the lift has 960 vertices
    
    def build_lift(d, voltages):
        """Build the d-cover of Petersen graph with given voltages."""
        n = 10 * d
        edges = []
        for i, (u, v) in enumerate(petersen_edges):
            a = voltages[i] % d
            for s in range(d):
                su = u * d + s
                sv = v * d + (s + a) % d
                edges.append((su, sv))
        return n, edges
    
    def compute_girth(n, edges):
        """Compute girth via BFS from each vertex, non-backtracking."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_girth = float('inf')
        
        for start in range(n):
            visited = {}
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited[(nb, start)] = 1
            
            while queue:
                node, prev, depth = queue.popleft()
                if node == start:
                    if depth < min_girth:
                        min_girth = depth
                    continue
                if depth >= min_girth:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if (nb, node) not in visited:
                        visited[(nb, node)] = depth + 1
                        queue.append((nb, node, depth + 1))
        
        return min_girth if min_girth != float('inf') else 0
    
    # Try different cover degrees and random voltage assignments
    # d must be such that 10*d is reasonable
    # Start with d values that give n in range [512, 2000]
    # d=52 -> n=520, d=60 -> n=600, d=72 -> n=720, d=96 -> n=960
    
    best_result = None
    
    for d in [52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120]:
        if time.time() > deadline:
            break
        
        n = 10 * d
        max_trials = 200
        found = False
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Random voltage assignment
            voltages = [random.randint(0, d - 1) for _ in range(15)]
            
            n_lift, edges = build_lift(d, voltages)
            girth = compute_girth(n_lift, edges)
            
            if girth >= g:
                return n_lift, edges
            
            if best_result is None or n_lift < best_result[0]:
                best_result = (n_lift, edges, girth)
        
        if found:
            break
    
    # If voltage graph didn't work, try Cayley graph approach
    # SL(2, q) has order q(q^2-1)
    # For q=16: 16*255 = 4080 (too large)
    # For q=8: 8*63 = 504 (below lower bound)
    # For q=9: 9*80 = 720
    # Let's try SL(2,9) or similar
    
    # Actually, let's try a different approach: use the known fact that
    # random regular graphs of sufficient size have large girth with high probability
    # For n vertices, girth g, the probability is high when n >> (k-1)^(g/2)
    
    # Fallback: random 3-regular graph search
    def generate_random_3_regular(n):
        if n < 4 or n * 3 % 2 != 0:
            return None
        
        half_edges = list(range(n * 3))
        vertices = list(range(n)) * 3
        random.shuffle(half_edges)
        
        edges = []
        for i in range(0, n * 3, 2):
            u = vertices[half_edges[i]]
            v = vertices[half_edges[i + 1]]
            if u != v:
                edges.append((u, v))
        
        deg = [0] * n
        seen = set()
        for u, v in edges:
            if u == v:
                return None
            key = (min(u, v), max(u, v))
            if key in seen:
                return None
            seen.add(key)
            deg[u] += 1
            deg[v] += 1
        if not all(d == 3 for d in deg):
            return None
        
        return edges
    
    for n in range(512, 1500, 4):
        if time.time() > deadline:
            break
        
        for _ in range(30):
            if time.time() > deadline:
                break
            edges = generate_random_3_regular(n)
            if edges is None:
                continue
            girth = compute_girth(n, edges)
            if girth >= g:
                return n, edges
    
    # Absolute fallback: LUW construction
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
