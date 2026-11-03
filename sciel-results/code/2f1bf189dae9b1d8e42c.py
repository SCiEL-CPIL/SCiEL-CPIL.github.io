import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # LUW Construction
    q = k
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return x * y % q

    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Precompute rule arrays for faster access
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rule_a[c], rule_b[c] = rule[c]
    
    def make_key(side, x):
        # Use integer encoding for faster hashing
        # side is 0 or 1, x is a tuple of d+1 elements
        # Encode as a single integer if possible, else use tuple
        return (side, x)
    
    # Start vertex
    start_x = (None,) + (0,) * d
    start_key = make_key(0, start_x)
    
    index = {start_key: 0}
    order = [(0, start_x)]
    edges = []
    
    # BFS with queue
    queue = [0]
    head = 0
    n_vertices = 1
    
    while head < len(order):
        if time.time() > deadline:
            break
        
        v_idx = queue[head]
        head += 1
        side, x = order[v_idx]
        
        for first in range(q):
            # Build y efficiently
            y_list = [None] * (d + 1)
            y_list[0] = None
            y_list[1] = first
            for c in range(2, d + 1):
                a = rule_a[c]
                b = rule_b[c]
                if side == 0:
                    y_list[c] = gf_add(x[c], gf_mul(y_list[a], x[b]))
                else:
                    y_list[c] = gf_sub(x[c], gf_mul(x[a], y_list[b]))
            
            other_side = 1 - side
            other_x = tuple(y_list)
            other_key = make_key(other_side, other_x)
            
            if other_key not in index:
                index[other_key] = n_vertices
                order.append((other_side, other_x))
                queue.append(n_vertices)
                n_vertices += 1
            
            if side == 0:
                edges.append((v_idx, index[other_key]))
                
    return n_vertices, edges
# EVOLVE-BLOCK-END
