"""Construct 3-regular graph of girth 16 using Lazebnik-Ustimenko-Woldar construction."""
import time
import numpy as np

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
    powers = np.array([1] * (d + 1), dtype=np.int64)
    for i in range(1, d + 1):
        powers[i] = powers[i-1] * q
    
    # Precompute digits for all vertices
    # digits[vid, pos] = digit at position pos (1-indexed, so index pos-1)
    num_vertices_side = q ** d
    all_ids = np.arange(num_vertices_side, dtype=np.int64)
    
    # Extract digits: for each vertex id, compute digit at each position
    # digit at position pos (1-indexed) = (id // q^(pos-1)) % q
    digits = np.zeros((num_vertices_side, d), dtype=np.int8)
    for pos in range(1, d + 1):
        digits[:, pos - 1] = (all_ids // powers[pos - 1]) % q
    
    def get_point_neighbours_array(p_id):
        # Point p, find Lines l such that l[c] = p[c] + l[a] * p[b]
        # For each possible l[1] (q choices), compute the full line id
        neighbors = []
        p_digits = digits[p_id]
        for l1 in range(q):
            current_l_id = l1
            l_digits = np.zeros(d, dtype=np.int8)
            l_digits[0] = l1
            for c in range(2, d + 1):
                a, b = rule[c]
                p_c = p_digits[c - 1]
                l_a = l_digits[a - 1]
                p_b = p_digits[b - 1]
                l_c = (p_c + l_a * p_b) % q
                l_digits[c - 1] = l_c
                current_l_id += l_c * powers[c - 1]
            neighbors.append(current_l_id)
        return neighbors

    def get_line_neighbours_array(l_id):
        # Line l, find Points p such that p[c] = l[c] - l[a] * p[b]
        # For each possible p[1] (q choices), compute the full point id
        neighbors = []
        l_digits = digits[l_id]
        for p1 in range(q):
            current_p_id = p1
            p_digits = np.zeros(d, dtype=np.int8)
            p_digits[0] = p1
            for c in range(2, d + 1):
                a, b = rule[c]
                l_c = l_digits[c - 1]
                l_a = l_digits[a - 1]
                p_b = p_digits[b - 1]
                p_c = (l_c - l_a * p_b) % q
                p_digits[c - 1] = p_c
                current_p_id += p_c * powers[c - 1]
            neighbors.append(current_p_id)
        return neighbors

    # BFS from start vertex (0, 0) which is Point 0
    # Vertex encoding: side * num_vertices_side + vid
    # side 0 = point, side 1 = line
    start_vertex = 0  # Point 0
    index = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        side = vertex // num_vertices_side
        vid = vertex % num_vertices_side
        
        if side == 0:
            neighbor_vids = get_point_neighbours_array(vid)
            n_side = 1
        else:
            neighbor_vids = get_line_neighbours_array(vid)
            n_side = 0
            
        for n_vid in neighbor_vids:
            n_vertex = n_side * num_vertices_side + n_vid
            if n_vertex not in index:
                index[n_vertex] = len(order)
                order.append(n_vertex)
            edges.append((index[vertex], index[n_vertex]))
                
    n = len(order)
    return n, edges
