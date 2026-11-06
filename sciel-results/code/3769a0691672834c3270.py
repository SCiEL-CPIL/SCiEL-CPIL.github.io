def construct():
    # Paley graph of order 17 plus an isolated vertex
    n = 18
    edges = []
    # Quadratic residues mod 17
    residues = {1, 2, 4, 8, 9, 13, 15, 16}
    
    for u in range(17):
        for v in range(u + 1, 17):
            diff = (v - u) % 17
            if diff in residues:
                edges.append([u, v])
                
    # Vertex 17 is isolated, so no edges involving 17
    
    return edges
