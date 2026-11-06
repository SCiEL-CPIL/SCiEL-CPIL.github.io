def construct():
    n = 16
    # Partition A = {0..5}, B = {6..15}
    # Edges: All cross edges (6*10=60)
    # Plus 5 internal edges in B forming a matching
    edges = []
    
    # Cross edges
    for u in range(6):
        for v in range(6, 16):
            edges.append([u, v])
            
    # Internal edges in B: (6,7), (8,9), (10,11), (12,13), (14,15)
    internal_b = [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    for u, v in internal_b:
        edges.append([u, v])
        
    return {'n': n, 'edges': edges}
