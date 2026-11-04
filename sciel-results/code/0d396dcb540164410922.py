"""Improved LUW construction with numpy vectorization and deadline guard."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Precompute rule table
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # GF(4) multiplication table
    if q == 4:
        gf4_table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64)
    else:
        gf4_table = None
    
    # BFS to find connected component
    start_vertex = (0, (0,) * (d + 1))
    visited = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    vertex_idx = 1
    for i, vertex in enumerate(order):
        if time.time() > deadline - 5.0:
            break
        
        side, x_tuple = vertex
        x = np.array(x_tuple, dtype=np.int64)
        
        # Generate all q neighbors at once using vectorized operations
        # Y has shape (q, d+1), Y[:, 1] = 0..q-1
        Y = np.zeros((q, d + 1), dtype=np.int64)
        Y[:, 1] = np.arange(q)
        
        for c in range(2, d + 1):
            a, b = rule[c]
            if side == 0:
                # Point -> Line: y[c] = x[c] + y[a] * x[b]
                if q == 4:
                    prod = gf4_table[Y[:, a], x[b]]
                    Y[:, c] = prod ^ x[c]
                else:
                    prod = (Y[:, a] * x[b]) % q
                    Y[:, c] = (prod + x[c]) % q
            else:
                # Line -> Point: y[c] = x[c] - x[a] * y[b]
                if q == 4:
                    prod = gf4_table[x[a], Y[:, b]]
                    Y[:, c] = prod ^ x[c]
                else:
                    prod = (x[a] * Y[:, b]) % q
                    Y[:, c] = (x[c] - prod) % q
        
        for j in range(q):
            neighbor = (1 - side, tuple(Y[j]))
            
            if neighbor not in visited:
                visited[neighbor] = vertex_idx
                order.append(neighbor)
                vertex_idx += 1
            
            if side == 0:
                edges.append((visited[vertex], visited[neighbor]))
    
    return len(order), edges
# EVOLVE-BLOCK-END
