import numpy as np

def ring_routing() -> np.ndarray:
    N = 10
    D = 15
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2)
    ]
    
    # Precompute edge lists for CW and CCW for each demand
    cw_edges = []
    ccw_edges = []
    weights = []
    
    for s, t, w in demands:
        cw_e = []
        curr = s
        while curr != t:
            cw_e.append(curr)
            curr = (curr + 1) % N
        cw_edges.append(cw_e)
        
        ccw_e = []
        curr = s
        while curr != t:
            curr = (curr - 1) % N
            # The edge traversed is the one from curr to curr+1? 
            # No, the edge index is usually defined by the start node in the CW direction.
            # Let's re-read carefully: "edge e joins nodes e and (e + 1) mod 10".
            # So edge index e connects node e and node e+1.
            # If I move from node u to node v counter-clockwise, I am traversing edge v-1 (mod 10)?
            # Let's trace:
            # Node 0 to Node 9 CCW.
            # Path: 0 -> 9.
            # Edge connecting 9 and 0 is edge 9 (joins 9 and 0).
            # So if I go from 0 to 9 CCW, I use edge 9.
            # Let's check my loop above.
            # curr starts at s.
            # curr = (curr - 1) % N.
            # The edge I just crossed connects curr+1 and curr.
            # The edge index is curr (since edge curr joins curr and curr+1).
            # So yes, append curr.
            
            ccw_e.append(curr)
        ccw_edges.append(ccw_e)
        weights.append(w)

    best_max_load = float('inf')
    best_mask = 0
    total_masks = 1 << D
    
    # To speed up, we can use numpy or simple lists.
    # Given 32k iterations, pure python is fine.
    
    for mask in range(total_masks):
        loads = [0] * N
        # Iterate through demands
        for i in range(D):
            w = weights[i]
            if (mask >> i) & 1:
                # CCW
                for e in ccw_edges[i]:
                    loads[e] += w
            else:
                # CW
                for e in cw_edges[i]:
                    loads[e] += w
        
        current_max = max(loads)
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = mask

    # Construct result
    routing = np.zeros(D)
    for i in range(D):
        if (best_mask >> i) & 1:
            routing[i] = 1.0 # CCW
        else:
            routing[i] = 0.0 # CW
            
    return routing
