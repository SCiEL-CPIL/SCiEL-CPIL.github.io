import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # 1. Determine minimal d for LUW construction CD(d, k)
    # Girth of CD(d, q) is 2d + 2.
    # We need girth >= g.
    # 2d + 2 >= g  =>  2d >= g - 2  =>  d >= (g - 2) / 2
    # d must be an integer >= 2.
    
    d = max(2, (g - 2 + 1) // 2)  # Ceiling division for (g-2)/2
    
    # Example: g=9 -> (7+1)//2 = 4.
    # Example: g=10 -> (8+1)//2 = 4 (since 9//2=4? No. (10-2+1)//2 = 9//2 = 4).
    # Example: g=11 -> (11-2+1)//2 = 10//2 = 5.
    # Example: g=12 -> (12-2+1)//2 = 11//2 = 5.
    
    q = k
    
    # 2. Field operations for GF(q)
    if q == 4:
        add_table = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def add(x, y): return add_table[x][y]
        def sub(x, y): return add_table[x][y] # char 2
        def mul(x, y): return mul_table[x][y]
    else:
        # Assume prime field
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # 3. Define the LUW rules
    rule = [None] * (d + 1)
    if d >= 2:
        rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
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

    # 4. Construct the graph via BFS from the zero vertex
    start_vertex = (0, (0,) * (d + 1))
    
    index = {start_vertex: 0}
    queue = [start_vertex]
    edges = []
    order = [start_vertex]
    
    for vertex in queue:
        if time.time() > deadline:
            break
            
        side, x = vertex
        v_idx = index[vertex]
        
        # Generate all q neighbours
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            
            if side == 0:
                # x is Point (p), y is Line (l)
                # l[c] = p[c] + l[a] * p[b]
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                # x is Line (l), y is Point (p)
                # p[c] = l[c] - l[a] * p[b]
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            neighbor = (1 - side, tuple(y))
            
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
                queue.append(neighbor)
            
            # Add edge only once. Graph is bipartite (side 0 vs side 1).
            # Add edge when current vertex is on side 0.
            if side == 0:
                edges.append((v_idx, index[neighbor]))

    n = len(order)
    
    return n, edges
