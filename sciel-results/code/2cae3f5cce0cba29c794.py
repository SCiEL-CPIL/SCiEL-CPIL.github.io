# EVOLVE-BLOCK-START
import time
import itertools

def construct(k, g):
    deadline = time.time() + 115
    q = k
    
    # Field operations for F_q
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    def generate_graph(d):
        n = 2 * (q ** d)
        q_pows = [1] * d
        for i in range(d - 1, 0, -1):
            q_pows[i] = q_pows[i + 1] * q
        
        def vertex_to_idx(side, x):
            idx = side * (q ** d)
            for i in range(d):
                idx += x[i] * q_pows[i]
            return idx

        edges = []
        
        # Define rules
        rules = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
            rules[c] = options[r]
            
        for side in range(2):
            for x_tuple in itertools.product(range(q), repeat=d):
                x = list(x_tuple)
                x.insert(0, 0)
                idx_u = vertex_to_idx(side, x[1:])
                
                for first in range(q):
                    y = [0] * (d + 1)
                    y[0] = 0
                    y[1] = first
                    
                    if side == 0:
                        for c in range(2, d + 1):
                            a, b = rules[c]
                            y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        for c in range(2, d + 1):
                            a, b = rules[c]
                            y[c] = sub(x[c], mul(x[a], y[b]))
                            
                    idx_v = vertex_to_idx(1 - side, y[1:])
                    edges.append((idx_u, idx_v))
                    
        return n, edges

    # Estimate minimum d
    # Girth of LUW is approx 2d+2
    min_d = max(3, (g - 2) // 2)
    
    for d in range(min_d, 10):
        if time.time() > deadline - 5:
            break
        n, edges = generate_graph(d)
        # Optional: quick check if n is reasonable
        return n, edges

    # Fallback
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n, edges = generate_graph(d)
    return n, edges
# EVOLVE-BLOCK-END
