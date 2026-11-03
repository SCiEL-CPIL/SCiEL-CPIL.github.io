import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # LUW Construction for k=4
    # We need girth >= g
    # Girth of LUW(4, d) is 2d + 2
    # So 2d + 2 >= g => d >= (g - 2) / 2
    # For g=10, d >= 4.
    # n = 2 * 4^d
    # For d=4, n = 512.
    
    d = (g - 2) // 2
    if d < 2:
        d = 2 # Safety, though g>=10 implies d>=4
        
    q = 4
    
    # GF(4) operations
    # 0:00, 1:01, 2:10, 3:11
    # Primitive poly x^2+x+1
    table = [[0, 0, 0, 0], 
             [0, 1, 2, 3], 
             [0, 2, 3, 1], 
             [0, 3, 1, 2]]
    
    def add(x, y): return x ^ y
    def sub(x, y): return x ^ y
    def mul(x, y): return table[x][y]
    
    # Recurrence rules for y_c = x_c + y_a * x_b
    # c=2: a=1, b=1
    # c=3: a=2, b=1
    # c=4: a=1, b=2
    # For c > 4, follow the pattern from the failed code if needed, 
    # but for g=10, d=4 is enough.
    # If d > 4, we need more rules.
    # The pattern in failed code:
    # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][(c - 5) % 4]
    # where c0 = c - (c - 5) % 4
    
    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rules[c] = (1, c0 - 2)
        elif r == 1:
            rules[c] = (c0 - 1, 1)
        elif r == 2:
            rules[c] = (c0, 1)
        else:
            rules[c] = (1, c0 + 1)
            
    def compute_y(x, j):
        # x is a tuple of length d (indices 1..d mapped to 0..d-1 in tuple, but let's use 1-based logic internally)
        # Let's store x as 0-indexed tuple: x[0] is x_1, x[1] is x_2, etc.
        # y[0] = j
        # y[c] for c=2..d (1-based)
        # y_idx = c-1
        
        y = [0] * d
        y[0] = j # y_1
        
        for c in range(2, d + 1):
            a, b = rules[c]
            # x_a is x[a-1], x_b is x[b-1]
            # y_a is y[a-1]
            # y_c is y[c-1]
            
            # Note: a and b are 1-based indices in the formula
            # If a=1, y_a = y[0]
            # If b=1, x_b = x[0]
            
            val = add(x[c-1], mul(y[a-1], x[b-1]))
            y[c-1] = val
            
        return tuple(y)
        
    edges = []
    n_vertices = 2 * (q ** d)
    
    # Iterate over all x on side 0
    for x_val in range(q ** d):
        # Convert x_val to tuple x
        x = [0] * d
        temp = x_val
        for i in range(d - 1, -1, -1):
            x[i] = temp % q
            temp //= q
            
        idx_0 = x_val # side 0, index x_val
        
        for j in range(q):
            y = compute_y(x, j)
            
            # Convert y to index
            y_val = 0
            for i in range(d):
                y_val = y_val * q + y[i]
                
            idx_1 = (q ** d) + y_val # side 1
            
            edges.append((idx_0, idx_1))
            
    n = n_vertices
    
    return n, edges
