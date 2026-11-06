def construct():
    # 1-factorization of K16: for vertex i in 0..15, use round-based matching
    # Standard construction: for n=16 (even), use 15 rounds
    # In round r (0..14), vertex 15 is matched with vertex r
    # Remaining 14 vertices {0..14} \ {r} are matched in pairs
    # Using the standard "circle method"
    
    colours = [0] * 120
    
    # Edge index: for pair (i,j) with i<j, index = i*15 + (j-i-1) 
    # Wait, let me compute: pairs (i,j), 0<=i<j<16
    # Number of pairs before (i,j): sum_{k=0}^{i-1} (15-k) + (j-i-1)
    # = sum_{k=0}^{i-1}(15-k) + j - i - 1
    # = i*15 - i*(i-1)//2 + j - i - 1
    
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        # count edges (k,l) with k < i: for each k from 0 to i-1, there are 15-k edges
        # that's sum_{k=0}^{i-1}(15-k) = i*15 - i*(i-1)//2
        # then edges (i, l) with i < l < j: that's j - i - 1
        return i * 15 - i * (i - 1) // 2 + (j - i - 1)
    
    n = 16
    # 1-factorization using circle method for even n
    # Fix vertex n-1, rotate the other n-1 vertices
    # For round r = 0, 1, ..., n-2:
    #   Match vertex n-1 with vertex r
    #   For the remaining vertices, use the standard pairing
    
    for r in range(n - 1):
        # Round r uses colour r
        # Vertex (n-1) matched with vertex r
        colours[edge_idx(n-1, r)] = r
        
        # Remaining vertices: all except (n-1) and r
        # Use the circle method on vertices 0..n-2
        # In round r, the pairing of {0,...,n-2}\{r} is:
        # For k = 1 to (n-2)/2:
        #   vertex ((r + k) mod (n-1)) matched with vertex ((r - k) mod (n-1))
        
        m = n - 1  # 15
        for k in range(1, m // 2 + 1):
            u = (r + k) % m
            v = (r - k) % m
            colours[edge_idx(u, v)] = r
    
    return colours
