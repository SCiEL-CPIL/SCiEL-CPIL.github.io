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
    next_id = [0]
    
    def get_id(tree_id, level, index):
        key = (tree_id, level, index)
        if key not in id_map:
            id_map[key] = next_id[0]
            next_id[0] += 1
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
    
    # The simple Moore bound construction for k=3, g=16 creates a graph with
    # girth 8 (cycles of length 8 exist via the connections between S0 and S1).
    # Specifically, for any i, s0[i]-s1[i]-s1[i+1]-s0[i+1]-s0[i] is a 4-cycle
    # if we consider the structure, or more accurately, the non-backtracking
    # paths from s0[i] to s0[i+1] via s1 form short cycles.
    
    # To get girth 16, we need a more sophisticated construction or to use
    # the LUW fallback which is known to produce higher girth graphs for these
    # parameters, although it might have more vertices.
    # The task asks for "as few vertices as possible".
    # The Moore bound for k=3, g=16 is 2*(3^8 - 1)/(3-1) = 2*(6561-1)/2 = 6560? No.
    # Moore bound M(k,g) for even g=2R is 2 * (k-1)^R.
    # For k=3, R=8: 2 * 2^8 = 512.
    # The program above constructs 2*(2^8 - 1) = 510 vertices?
    # Wait, sum_{i=0}^{R-1} 2^i = 2^R - 1.
    # Two trees: 2 * (2^R - 1) = 2 * 255 = 510.
    # But the Moore bound is 512.
    # The construction is valid for girth g IF the connections between the
    # two trees don't create shorter cycles.
    # For k=3, g=16, the simple "double tree" construction (Hoffman-Singleton style
    # but for trees) typically yields girth g only for specific small cases.
    # For g=16, the cycle length created by connecting the leaves is often much shorter.
    
    # Let's check the cycle length in the constructed graph.
    # Path from s0[i] to s1[i] is length 1.
    # Path from s1[i] to s0[i+1] is length 1.
    # Path from s0[i+1] to s0[i] in T0:
    # s0[i] and s0[i+1] are at level R-1.
    # Their LCA in T0 is at level 0 (root 0) if i and i+1 are far apart, or higher.
    # Actually, s0[i] corresponds to index i in level R-1.
    # The path between two leaves in a binary tree of depth R-1 is at most 2*(R-1).
    # So a cycle could be s0[i]-s1[i]-s1[i+1]-s0[i+1]-...-s0[i].
    # Length = 1 + 1 + dist_T0(s0[i+1], s0[i]).
    # dist_T0(s0[i], s0[i+1]) can be small.
    # If i and i+1 are siblings? No, they are at the same level.
    # If i=0, i+1=1.
    # s0[0] is leftmost leaf. s0[1] is second leftmost.
    # Their LCA is at level R-2.
    # dist = 2.
    # Cycle: s0[0]-s1[0]-s1[1]-s0[1]-s0[0]. Length 4.
    # This violates girth 16.
    
    # Therefore, the simple construction is INVALID for girth 16.
    # We must use the LUW fallback or another valid construction.
    # The LUW fallback in the code is a known heuristic that often produces
    # valid cages or near-cages.
    
    return _luw_construct(k, g)

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
