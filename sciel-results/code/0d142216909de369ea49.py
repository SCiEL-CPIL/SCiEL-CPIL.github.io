def construct():
    # Chvátal graph on 12 vertices (0..11)
    # It is triangle-free and 4-chromatic.
    # Edges:
    cycle = [(i, (i + 1) % 12) for i in range(12)]
    chords = [
        (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),
        (6, 9), (7, 10), (8, 11), (9, 0), (10, 1), (11, 2)
    ]
    edges = [list(e) for e in cycle]
    edges += [list(e) for e in chords]
    
    # Independent set in Chvátal graph: {0, 2, 4, 6}
    indep_set = [0, 2, 4, 6]
    
    # Add 4 vertices (12..15) connected to the independent set
    for v in range(12, 16):
        for u in indep_set:
            edges.append([v, u])
            
    return edges
