import time

def construct():
    # 1. Define Petersen Graph
    # Vertices 0..9
    # Edges:
    # Outer cycle: (0,1), (1,2), (2,3), (3,4), (4,0)
    # Inner star: (5,7), (7,9), (9,6), (6,8), (8,5)
    # Spokes: (0,5), (1,6), (2,7), (3,8), (4,9)
    
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    def mycieski(edges, n):
        # Returns new edges and new n
        new_edges = []
        # 1. Original edges
        for u, v in edges:
            new_edges.append((u, v))
        
        # 2. Spokes (i, n+i)
        for i in range(n):
            new_edges.append((i, n + i))
        
        # 3. Shadow edges (n+i, n+j) if (i,j) in edges
        for u, v in edges:
            new_edges.append((n + u, n + v))
            
        return new_edges, 2 * n + 1

    # Apply Mycieski twice
    # Step 1: M(Petersen)
    edges1, n1 = mycieski(petersen_edges, 10)
    # Step 2: M(M(Petersen))
    edges2, n2 = mycieski(edges1, n1)
    
    # n2 should be 43
    # We want n=40
    # Take subgraph induced by vertices 0..39
    target_n = 40
    final_edges = []
    for u, v in edges2:
        if u < target_n and v < target_n:
            final_edges.append([u, v])
            
    return {'n': target_n, 'edges': final_edges}
