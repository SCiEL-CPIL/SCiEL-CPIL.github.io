def construct():
    edges = []
    # Part A: 0, 1, 2
    # Part B: 3..15
    
    # 1. Complete bipartite K(3, 13)
    for a in range(3):
        for b in range(3, 16):
            edges.append([a, b])
            
    # 2. Triangle in A
    edges.append([0, 1])
    edges.append([1, 2])
    edges.append([0, 2])
    
    # 3. One edge in B
    edges.append([3, 4])
    
    return edges
