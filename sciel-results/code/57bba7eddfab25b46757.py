def construct():
    # Maximum edges in a triangle-free graph on 16 vertices is 64 (Mantel's theorem).
    # K_{8,8} achieves this bound, has diameter 2, and every edge is essential:
    # removing any edge disconnects its two endpoints (diameter becomes infinity).
    
    part_a = [i for i in range(8)]
    part_b = [i for i in range(8, 16)]
    
    edges = []
    for a in part_a:
        for b in part_b:
            edges.append([a, b])
    
    return edges
