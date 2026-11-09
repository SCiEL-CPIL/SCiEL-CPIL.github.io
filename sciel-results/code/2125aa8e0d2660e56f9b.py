import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # For k=3, g=16: use q=3, d=11 (girth >= d+5 = 16)
    q = k
    # d must be odd and >= g-5
    if (g - 5) % 2 == 0:
        d = g - 4
    else:
        d = g - 5
    d = max(d, 3)
    
    # Precompute powers of q for encoding
    # Vector indices are 1..d. We map them to 0..d-1 for integer encoding.
    # Let vector v have components v[1]...v[d].
    # ID = sum(v[i] * q^(i-1) for i in 1..d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Build the rule
    # rule[c] = (a, b) such that y[c] depends on y[a] and y[b]
    # Indices in the vector are 1..d
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The pattern is (1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)
        # Note: c0 is a multiple of 4?
        # c=5: r=0, c0=5. (1, 3)
        # c=6: r=1, c0=4. (3, 1) -> Wait, c0-1=3.
        # c=7: r=2, c0=4. (4, 1)
        # c=8: r=3, c0=4. (1, 5)
        # c=9: r=0, c0=9. (1, 7)
        # This matches the standard LUW definition.
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
    # We represent a vertex by (side, id)
    # side 0: Point, side 1: Line
    # id is the integer encoding of the vector of length d
    
    # Total possible IDs: q^d
    # We need to extract digits from the ID.
    # To speed up, we can precompute the digits for all IDs?
    # q^d for q=3, d=11 is 177147. This is small enough to precompute digits.
    # But we only visit a subset.
    
    # Let's use a dictionary for visited vertices: {(side, id): index}
    # And a list of vertices in BFS order.
    
    # To get the digit at position i (1-indexed) from the integer ID:
    # digit = (id // q^(i-1)) % q
    
    # Precompute powers
    powers = [1] * (d + 1)
    for i in range(1, d + 1):
        powers[i] = powers[i-1] * q
        
    def get_digit(val, pos):
        return (val // powers[pos-1]) % q
        
    def set_digit(val, pos, digit):
        # This is not efficient if we do it repeatedly.
        # Better to build the ID from scratch in the neighbor generation.
        pass

    # Neighbor generation for a Point (side=0) with ID p_id:
    # We need to find Line IDs l_id such that for all c=2..d:
    # l[c] = p[c] + l[a] * p[b]
    # l[1] is free (0..q-1).
    
    def get_point_neighbours(p_id):
        # Extract p digits
        # p_digits[1..d]
        # We iterate over l[1] (first)
        for l1 in range(q):
            l_id = l1 # l[1] is the least significant digit? 
            # My encoding: ID = sum v[i] * q^(i-1). So v[1] is LSB.
            # So l_id starts with l1.
            
            # We need to compute l[2]..l[d]
            # We store computed digits in a list or update l_id
            # l_id is built as we go.
            # l_id = l1
            # For c from 2 to d:
            #   a, b = rule[c]
            #   l_c = p_c + l_a * p_b
            #   l_id += l_c * powers[c-1]
            
            # We need p_c, l_a, p_b
            # p_c = get_digit(p_id, c)
            # l_a = get_digit(l_id, a)  # Note: l_id is being built, so l_a is already in l_id
            # p_b = get_digit(p_id, b)
            
            current_l_id = l1
            valid = True
            for c in range(2, d + 1):
                a, b = rule[c]
                p_c = get_digit(p_id, c)
                l_a = get_digit(current_l_id, a)
                p_b = get_digit(p_id, b)
                l_c = add(p_c, mul(l_a, p_b))
                current_l_id += l_c * powers[c-1]
            
            yield (1, current_l_id)

    def get_line_neighbours(l_id):
        # Line l, find Points p such that:
        # l[c] = p[c] + l[a] * p[b]  =>  p[c] = l[c] - l[a] * p[b]
        # p[1] is free.
        for p1 in range(q):
            current_p_id = p1
            for c in range(2, d + 1):
                a, b = rule[c]
                l_c = get_digit(l_id, c)
                l_a = get_digit(l_id, a)
                p_b = get_digit(current_p_id, b)
                p_c = sub(l_c, mul(l_a, p_b))
                current_p_id += p_c * powers[c-1]
            yield (0, current_p_id)

    # BFS
    start_vertex = (0, 0) # Point 0
    index = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        side, vid = vertex
        if side == 0:
            neighbors = get_point_neighbours(vid)
        else:
            neighbors = get_line_neighbours(vid)
            
        for n_side, n_vid in neighbors:
            n_vertex = (n_side, n_vid)
            if n_vertex not in index:
                index[n_vertex] = len(order)
                order.append(n_vertex)
            if side == 0:
                edges.append((index[vertex], index[n_vertex]))
                
    n = len(order)
    return n, edges
