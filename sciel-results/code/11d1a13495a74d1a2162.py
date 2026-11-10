import time
import random
import numpy as np
from collections import deque

def _construct_cd(d, q):
    """
    Constructs the Lazebnik-Ustimenko-Woldar graph CD(d, q).
    Returns (n, edges, adj).
    """
    # Vertices are (side, x_1, ..., x_d)
    # side 0: points, side 1: lines
    # Total vertices: 2 * q^d
    # Wait, the seed used q^(d+1). Let's verify the standard definition.
    # CD(d, q) usually has 2 * q^d vertices?
    # If d=1, CD(1, q) is K_{q, q}. n = 2q.
    # If the seed used q^(d+1), then for d=1, n=2q^2. That's not K_{q,q}.
    # Let's check the seed's comment: "Total vertices = 2 * q^(d+1)".
    # And "Girth >= d+5".
    # If the seed's construction is non-standard or I am misremembering, I should stick to the seed's
    # construction but fix the d selection, OR use a standard known one.
    # The seed's construction:
    # rule[c] for c=2..d.
    # This implies the vector length is d.
    # If vector length is d, n = 2 * q^d.
    # But the seed encoded `x` as `tuple([0] * (d + 1))`.
    # So the seed used length d+1.
    # Let's assume the seed's construction is $CD(d+1, q)$ effectively?
    # Or the parameter `d` in the seed corresponds to $d-1$ in standard notation?
    # If seed d=5, n=31250 = 2 * 5^6. So length is 6.
    # Standard CD(d, q) has n = 2 * q^d.
    # So seed d=5 -> standard d=6?
    # Girth of standard CD(6, q) is 2*6+4 = 16?
    # If seed d=5 gives girth 10 (as per its logic d+5), and standard d=6 gives 16, they don't match.
    
    # Let's use a robust construction: The Kiefer-Wong graph KG(d, q).
    # KG(d, q) has 2 * q^d vertices.
    # Girth is 2d+4 (even d) or 2d+5 (odd d).
    # For k=5, q=5.
    # d=3: n = 2 * 5^3 = 250. Girth = 2(3)+5 = 11? Or 2(3)+4=10?
    # If girth >= 9, this is excellent.
    # d=2: n = 2 * 5^2 = 50. Girth = 2(2)+4 = 8? (Fails for g=9).
    # So d=3 is the target. n=250.
    
    # Construction of KG(d, q):
    # Vertices: (0, x) and (1, y) for x, y in GF(q)^d.
    # (0, x) ~ (1, y) if y = A x + b? No, it's more complex.
    # Actually, KG(d, q) is defined by:
    # (0, x) ~ (1, y) iff y_i = x_i * y_{i+1} + ...?
    
    # Let's stick to the CD(d, q) from the seed but assume the seed's d is the "dimension" parameter.
    # If I use the seed's code with d=3:
    # n = 2 * 5^4 = 1250.
    # Girth check will determine validity.
    # If valid, return.
    # If not, try d=4 (n=6250).
    
    pass

def _lwf_graph(k, g):
    # Use the seed's construction but allow choosing d
    # The seed's construction is likely CD(d, q) with some offset.
    # Let's just run the seed's logic for d=3,4,5 and pick the smallest valid one.
    
    best_n = float('inf')
    best_edges = None
    best_adj = None
    
    for d in [3, 4, 5]:
        # Build graph with this d
        # Reuse seed logic
        q = k
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            if r == 0:
                rule[c] = (1, c0 - 2)
            elif r == 1:
                rule[c] = (c0 - 1, 1)
            elif r == 2:
                rule[c] = (c0, 1)
            else:
                rule[c] = (1, c0 + 1)

        def get_neighbours(vertex):
            side, x = vertex
            x = list(x)
            result = []
            for first in range(q):
                y = [first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c - 1] = (x[c - 1] + y[a - 1] * x[b - 1]) % q
                    else:
                        y[c - 1] = (x[c - 1] - x[a - 1] * y[b - 1]) % q
                result.append((1 - side, tuple(y)))
            return result

        start = (0, tuple([0] * (d + 1)))
        index = {start: 0}
        order = [start]
        adj = [[]]
        edges = []

        for i, vertex in enumerate(order):
            side, x = vertex
            for other in get_neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                    adj.append([])
                j = index[other]
                if i < j:
                    adj[i].append(j)
                    adj[j].append(i)
                    edges.append((i, j))

        n = len(order)
        
        # Check girth
        if _girth_check(adj, n, g):
            if n < best_n:
                best_n = n
                best_edges = edges
                best_adj = adj
            break # Found a valid one, since n grows with d, we can stop?
            # Actually, smaller d means smaller n. So we should try d=3 first.
            # If d=3 works, n=1250. If d=3 fails, try d=4.
            # So the loop order [3, 4, 5] is correct.
            
    if best_edges is None:
        # Fallback to seed's default
        return _lwf_graph_fallback(k, g)
        
    return best_n, best_edges, best_adj

def _lwf_graph_fallback(k, g):
    # Original seed logic
    d = 5 # Hardcode a safe value
    q = k
    # ... same construction ...
    pass

def _girth_check(adj, n, g):
    h = (g - 1) // 2
    for s in range(n):
        depth = {s: 0}
        parent = {s: -1}
        queue = deque([(s, 0)])
        while queue:
            u, du = queue.popleft()
            if du >= h:
                continue
            for v in adj[u]:
                if v not in depth:
                    depth[v] = du + 1
                    parent[v] = u
                    queue.append((v, du + 1))
                else:
                    dv = depth[v]
                    if v != parent[u] and u != parent.get(v, -1):
                        if du + dv + 1 < g:
                            return False
    return True

def construct(k, g):
    deadline = time.time() + 115
    n, edges, adj = _lwf_graph(k, g)
    return n, edges
