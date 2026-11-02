import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Parameters for k=3, g=16
    # Moore bound construction
    if k != 3 or g % 2 != 0:
        # Fallback to LUW if parameters don't match
        return _luw_construct(k, g)
        
    R = g // 2
    # Build tree T0 from 0
    # 0 is root. Neighbors 1, 2, 3.
    # 1 is the "bridge" vertex.
    # 2, 3 start the tree.
    
    # We will generate vertices and edges
    # Vertex 0
    # Vertex 1
    # Tree from 0 (excluding 1):
    # Level 0: 0
    # Level 1: 2, 3
    # Level 2: children of 2, 3
    # ...
    # Level R-1: children of Level R-2
    
    # Tree from 1 (excluding 0):
    # Level 0: 1
    # Level 1: 4, 5
    # Level 2: children of 4, 5
    # ...
    # Level R-1
    
    # Total vertices: 2 * sum_{i=0}^{R-1} 2^i = 2 * (2^R - 1)
    # For g=16, R=8. 2 * (256-1) = 510.
    
    n_vertices = 2 * (2**R - 1)
    
    # Mapping of vertex to (tree_id, level, index_in_level)
    # tree_id: 0 for T0, 1 for T1
    # level: 0..R-1
    # index: 0..2^level - 1
    
    # We need to assign actual IDs 0..n-1
    # Let's assign IDs sequentially.
    
    id_map = {}
    next_id = 0
    
    def get_id(tree_id, level, index):
        key = (tree_id, level, index)
        if key not in id_map:
            id_map[key] = next_id
            nonlocal next_id
            id_map[key] = next_id
            next_id += 1
        return id_map[key]
    
    # Initialize
    # T0
    get_id(0, 0, 0) # Vertex 0
    # T1
    get_id(1, 0, 0) # Vertex 1
    
    edges = []
    
    # Add edge 0-1
    edges.append((id_map[(0,0,0)], id_map[(1,0,0)]))
    
    # Build T0
    # Level 0 has 1 vertex (index 0)
    # Level 1 has 2 vertices (indices 0, 1)
    # ...
    
    for level in range(R):
        num_vertices_in_level = 2 ** level
        for idx in range(num_vertices_in_level):
            u_id = get_id(0, level, idx)
            if level < R - 1:
                # Connect to children in level+1
                # Each vertex has 2 children (since parent takes 1 slot, k=3)
                # Except root?
                # Root 0 has neighbors 1, 2, 3.
                # In T0, 0 is level 0.
                # Its children in T0 are 2, 3.
                # So level 0 vertex 0 connects to level 1 vertices 0, 1.
                # Level 1 vertex i connects to level 2 vertices 2i, 2i+1.
                
                child_1 = get_id(0, level + 1, 2 * idx)
                child_2 = get_id(0, level + 1, 2 * idx + 1)
                edges.append((u_id, child_1))
                edges.append((u_id, child_2))
                
                # Wait, for level 0, we already added edge to 1?
                # No, 1 is in T1.
                # So 0 connects to 1 (T1), 2 (T0), 3 (T0).
                # My loop adds edges to children in T0.
                # For level 0, idx 0:
                # Connects to child_1 (L1, 0), child_2 (L1, 1).
                # This covers 2 and 3.
                # And we manually added edge to 1.
                # So degree of 0 is 3. Correct.
                
                # For level > 0:
                # Parent is in level-1.
                # Children are 2.
                # Total degree 3. Correct.
                
            else:
                # Level R-1. No children in tree.
                pass
                
    # Build T1
    for level in range(R):
        num_vertices_in_level = 2 ** level
        for idx in range(num_vertices_in_level):
            u_id = get_id(1, level, idx)
            if level < R - 1:
                child_1 = get_id(1, level + 1, 2 * idx)
                child_2 = get_id(1, level + 1, 2 * idx + 1)
                edges.append((u_id, child_1))
                edges.append((u_id, child_2))
            else:
                pass
    
    # Now connect S0 and S1
    # S0: vertices in T0 at level R-1
    # S1: vertices in T1 at level R-1
    
    s0 = [get_id(0, R-1, i) for i in range(2**(R-1))]
    s1 = [get_id(1, R-1, i) for i in range(2**(R-1))]
    
    # Connect s0[i] to s1[i] and s1[(i+1)%len]
    n_s = len(s0)
    for i in range(n_s):
        edges.append((s0[i], s1[i]))
        edges.append((s0[i], s1[(i+1) % n_s]))
        
    return n_vertices, edges

def _luw_construct(k, g):
    # Fallback
    q = k
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
    return len(order), edges
