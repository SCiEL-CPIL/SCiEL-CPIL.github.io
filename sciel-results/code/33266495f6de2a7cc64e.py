def get_hs_graph():
    n = 50
    adj = [[False]*n for _ in range(n)]
    # Vertex 49 is the special one
    # Vertices 0..48 are (i, j) where i = v//7, j = v%7
    
    # Connect 49 to (i, 0)
    for i in range(7):
        v = i*7
        adj[49][v] = True
        adj[v][49] = True
        
    for i in range(7):
        for j in range(7):
            v = i*7 + j
            # Neighbors in the same row (cyclic)
            u1 = i*7 + (j-1)%7
            u2 = i*7 + (j+1)%7
            adj[v][u1] = True
            adj[u1][v] = True
            adj[v][u2] = True
            adj[u2][v] = True
            
            # Other neighbors?
            # The degree must be 7. We have 2 so far (plus 1 if j=0 from 49).
            # If j=0, deg is 3 so far. Need 4 more.
            # If j!=0, deg is 2 so far. Need 5 more.
            
            # The other neighbors are determined by the property that the graph is C4-free and regular.
            # Actually, the neighbors are (i+j, j) and (i-j, j) is NOT correct.
            
            # Correct formula:
            # Neighbors of (i, j) are:
            # (i, j-1), (i, j+1)
            # (i+j, j) ? No.
            
            # Let's check the neighbors of (0,0).
            # (0, 6), (0, 1)
            # And 49.
            # And 4 others.
            
            # The HS graph can be defined by:
            # (x, y) ~ (x', y') iff x' = x and y' = y +/- 1
            # OR x' = x + y and y' = y ? No.
            
            # Reference:
            # Neighbors of (i, j) are:
            # (i, j-1), (i, j+1)
            # (i+j, j) is not a vertex if we map to 0..48?
            
            # Let's use the algebraic definition:
            # V = Z_7 x Z_7 U {inf}
            # (x, y) ~ (x', y') if:
            # 1. x = x' and y - y' in {1, -1}
            # 2. x - x' = y - y' (mod 7) AND x != x' ? No.
            
            # Actually, the condition is:
            # (x, y) ~ (x', y') iff x' = x and y' = y +/- 1
            # OR x' = x + y and y' = y ?
            
            # Let's try this specific construction:
            # Neighbors of (i, j):
            # 1. (i, j-1)
            # 2. (i, j+1)
            # 3. (i+j, j) -- wait, if j=0, (i,0).
            # 4. (i-j, j) -- wait, if j=0, (i,0).
            
            # This construction is for the Clebsch graph? No.
