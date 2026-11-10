def construct():
    n = 16
    # Partition vertices into two sets of 8
    # A: 0-7, B: 8-15
    # Complete bipartite graph K_8,8
    # This graph is triangle-free.
    # Diameter is 2.
    # Every edge is critical because removing any edge (u,v) with u in A, v in B
    # increases the distance between u and v to 3 (via A-B-A or B-A-B paths don't exist directly, 
    # wait, u connects to B, v connects to A. Path u-b-a-v is length 3).
    # So diameter becomes 3.
    
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
            
    return edges
