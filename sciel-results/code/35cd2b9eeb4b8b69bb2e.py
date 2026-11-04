import numpy as np
import random
import time

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses algebraic construction based on finite fields (Moore graph style).
    """
    start_time = time.time()
    time_limit = 115.0

    # Algebraic construction for large girth regular graphs
    # The parameter q is the base of the finite field (must be a prime power)
    # We want q >= k for the degree to be at least k.
    # Actually, the construction produces a q-regular graph.
    # If we need degree k, we can either:
    # 1. Use q=k (if k is a prime power)
    # 2. Use q > k and delete edges? No, that breaks regularity.
    # 3. Use a different construction.
    
    # The seed program used q=k. This implies k must be a prime power.
    # For k=5, q=5 is a prime power.
    # The construction yields a graph with girth >= g.
    # Let's verify the girth condition.
    
    # The parameter d controls the "radius" of the construction.
    # The seed used d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # Let's analyze this.
    # For g=9: g-5=4, 4%2=0, so d = g-4 = 5.
    # For g=8: g-5=3, 3%2=1, so d = g-5 = 3.
    # For g=7: g-5=2, 2%2=0, so d = g-4 = 3.
    # For g=6: g-5=1, 1%2=1, so d = g-5 = 1 -> max(3,1)=3.
    
    # The construction is based on the "Moore graph" or "cage" construction.
    # The number of vertices is roughly 2 * q * q^d = 2 * q^(d+1).
    # For k=5, g=9, d=5: n ~ 2 * 5^6 = 31250. This is large.
    # The seed program returned 3125 vertices? No, the seed program's fallback
    # produced a graph, but the random search was the main part.
    # Wait, the seed program's fallback is what I'm looking at.
    # Let's check the size of the graph produced by the algebraic construction.
    
    # The construction generates vertices as (side, x) where x is a tuple of length d.
    # x[0] is None. x[1..d] are in [0, q).
    # So there are 2 * q^d possible vertices.
    # For k=5, g=9, d=5: n = 2 * 5^5 = 3125.
    # This matches the "seed gave 3125" comment in the original code.
    
    # So the algebraic construction gives n = 2 * q^d.
    # To minimize n, we want to minimize d.
    # The girth is guaranteed to be at least g if d is chosen correctly.
    # The seed's choice of d seems to ensure girth >= g.
    
    # Let's verify the girth for k=5, g=9, d=5.
    # The construction is a bipartite graph between two sets of q^d vertices.
    # Each vertex has degree q.
    # The girth is determined by the shortest cycle.
    # The algebraic structure ensures no short cycles.
    
    # Can we reduce d?
    # If we reduce d, the girth might drop below g.
    # The formula d = max(3, g - 5 if (g - 5) % 2 else g - 4) is a heuristic.
    # Let's try to find a smaller d that still guarantees girth >= g.
    
    # The girth of the constructed graph is related to d.
    # For the "Moore" construction, the girth is roughly 2d + 1 or 2d + 2.
    # If girth >= 2d + 1, then we need 2d + 1 >= g => d >= (g-1)/2.
    # For g=9, d >= 4.
    # The seed used d=5.
    # Let's try d=4.
    # If d=4, n = 2 * 5^4 = 1250.
    # Is the girth >= 9?
    # The construction is based on a linear recurrence.
    # The cycle length is related to the order of the matrix.
    # It's safer to use the seed's formula, but we can experiment.
    
    # However, we must ensure the graph is valid.
    # Let's stick to the seed's formula for safety, but check if we can improve it.
    
    # The task says the smallest known graph has 1116 vertices.
    # 1250 is closer to 1116 than 3125.
    # Let's try d = max(3, (g - 1) // 2).
    # For g=9, d=4.
    # For g=8, d=3.
    # For g=7, d=3.
    
    # Let's implement the construction with a variable d and check the girth.
    # If the girth is too small, we increase d.
    
    q = k
    # Ensure q is a prime power. For k=5, q=5.
    # If k is not a prime power, we might need to use a different q.
    # But the task says k=5. Let's assume k is a prime power.
    # If k is not a prime power, we can use the next prime power >= k.
    # But that would change the degree.
    # The construction produces a q-regular graph.
    # So we must have q=k.
    
    # Let's try different values of d and check the girth.
    # We want the smallest d such that girth >= g.
    
    def check_girth(adj, n, g):
        # BFS from each node to find shortest cycle
        min_cycle = float('inf')
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                if dist[u] >= g: # We only care about cycles < g
                    continue
                for v in np.where(adj[u])[0]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            if min_cycle < g:
                                return min_cycle
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # Try d from a minimum value upwards
    # Minimum d is 1, but we need d >= 3 for the construction to work properly?
    # The seed used max(3, ...). Let's start from 3.
    
    best_n = None
    best_edges = None
    best_d = None
    
    # Upper bound on d to avoid infinite loop
    max_d = 10
    
    for d in range(3, max_d + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Construct the graph with this d
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
        
        n = len(order)
        
        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edges:
            adj[u, v] = True
            adj[v, u] = True
        
        # Check girth
        girth = check_girth(adj, n, g)
        
        if girth >= g:
            best_n = n
            best_edges = edges
            best_d = d
            break # Found the smallest d that works
        
        # If girth < g, try a larger d
    
    if best_n is None:
        # Fallback to the seed's formula
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
        
        best_n = len(order)
        best_edges = edges

    return best_n, best_edges
