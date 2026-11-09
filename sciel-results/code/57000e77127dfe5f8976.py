N = 40

# EVOLVE-BLOCK-START
def construct():
    # Vertices:
    # u_i = i for i in 0..19
    # v_i = i + 20 for i in 0..19
    
    edges = []
    cycles = []
    
    # 1. Construct Edges
    # Rungs: (u_i, v_i)
    for i in range(20):
        edges.append((i, i + 20))
        
    # Top Rim: (u_i, u_{i+1})
    for i in range(19):
        edges.append((i, i + 1))
        
    # Bottom Rim: (v_i, v_{i+1})
    for i in range(19):
        edges.append((i + 20, i + 21))
        
    # Chord: (u_0, v_1) -> (0, 21)
    edges.append((0, 21))
    
    # 2. Construct Cycles
    # k from 3 to 40
    for k in range(3, N + 1):
        if k % 2 == 1:
            # Odd length k = 2j + 1
            j = (k - 1) // 2
            # Path: u_0 -> v_1 -> ... -> v_j -> u_j -> u_{j-1} -> ... -> u_1 -> u_0
            # Vertices:
            cycle = [0] # u_0
            # v_1 to v_j
            for i in range(1, j + 1):
                cycle.append(i + 20)
            # u_j down to u_1
            for i in range(j, 0, -1):
                cycle.append(i)
            # The cycle is closed by edge (u_1, u_0) which is (1,0) in rim?
            # Wait, last vertex is u_1. First is u_0.
            # Edge (1,0) exists in Top Rim.
            cycles.append(cycle)
        else:
            # Even length k = 2w + 2
            w = k // 2 - 1
            # Path: u_0 -> ... -> u_w -> v_w -> ... -> v_0 -> u_0
            cycle = []
            # u_0 to u_w
            for i in range(w + 1):
                cycle.append(i)
            # v_w down to v_0
            for i in range(w, -1, -1):
                cycle.append(i + 20)
            # Last vertex is v_0 (20). First is u_0 (0).
            # Edge (20, 0) is rung 0. Exists.
            cycles.append(cycle)
            
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
