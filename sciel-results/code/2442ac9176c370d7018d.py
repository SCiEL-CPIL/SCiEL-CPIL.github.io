import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    # For the specific case k=3, g=16, use LUW with q=3, d=11
    if k == 3 and g == 16:
        q = 3
        # d must be such that d + 5 >= g
        # For odd d, girth >= d + 5
        # 11 + 5 = 16
        d = 11
        return _luw_construct(q, d, deadline)
    
    # Fallback for other parameters (not expected in this task, but good for robustness)
    return _luw_construct(k, g - 5 if (g - 5) % 2 else g - 4, deadline)

def _luw_construct(q, d, deadline):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q).
    Returns (n, edges) for the connected component of the zero vertex.
    """
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
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
    # Precompute powers of q for encoding
    # ID = sum(v[i] * q^(i-1) for i in 1..d)
    powers = [1] * (d + 1)
    for i in range(1, d + 1):
        powers[i] = powers[i-1] * q
        
    def get_digit(val, pos):
        return (val // powers[pos-1]) % q

    def get_point_neighbours(p_id):
        # Point p, find Lines l
        # l[c] = p[c] + l[a] * p[b]
        # l[1] is free
        for l1 in range(q):
            current_l_id = l1
            for c in range(2, d + 1):
                a, b = rule[c]
                p_c = get_digit(p_id, c)
                l_a = get_digit(current_l_id, a)
                p_b = get_digit(p_id, b)
                l_c = add(p_c, mul(l_a, p_b))
                current_l_id += l_c * powers[c-1]
            yield (1, current_l_id)

    def get_line_neighbours(l_id):
        # Line l, find Points p
        # p[c] = l[c] - l[a] * p[b]
        # p[1] is free
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

    # BFS to find connected component
    start_vertex = (0, 0) # (side, id)
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
            # Add edge only once (when visiting from point side to avoid duplicates in undirected graph representation)
            # In LUW, edges connect Point to Line.
            # If we add edge when side==0, we cover all edges exactly once.
            if side == 0:
                edges.append((index[vertex], index[n_vertex]))
                
    n = len(order)
    return n, edges
