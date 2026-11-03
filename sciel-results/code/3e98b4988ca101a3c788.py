    def construct(k, g):
        q = k
        # Determine d such that girth >= g
        # Girth of LUW(d) is 2*d + 2
        # We need 2*d + 2 >= g
        # d >= (g - 2) / 2
        d = (g - 2 + 1) // 2 # Ceiling division?
        # If g=9, (7+1)//2 = 4.
        # If g=8, (6+1)//2 = 3.5 -> 4? No, (8-2)//2 = 3.
        # Let's just check:
        # d=2 -> g=6
        # d=3 -> g=8
        # d=4 -> g=10
        # So for g=9, we need d=4.
        # Formula: d = max(2, (g - 1) // 2) ?
        # g=9 -> (8)//2 = 4.
        # g=8 -> (7)//2 = 3.
        # g=6 -> (5)//2 = 2.
        # This works.
        d = max(2, (g - 1) // 2)
        
        # Precompute GF(q) tables
        add_t = [[(i+j)%q for j in range(q)] for i in range(q)]
        mul_t = [[(i*j)%q for j in range(q)] for i in range(q)]
        
        n_verts = 2 * (q ** d)
        edges = []
        
        # Iterate over Side 0 vertices
        # Index 0 to q^d - 1
        for idx in range(q ** d):
            # Decode coordinates x_1...x_d
            # Let's say x_1 is the most significant digit?
            # Or least?
            # Standard LUW formulas usually index 1 to d.
            # Let's use x[1]...x[d]
            coords = [0] * (d + 1) # 1-indexed
            temp = idx
            for i in range(1, d + 1):
                coords[i] = temp % q
                temp //= q
            
            # Compute neighbors on Side 1
            # y[1] = x[1]
            # y[i] = x[i] + sum_{j < i} x[j] * y[j]  (Note: standard LUW uses x[j]*y[j] or similar?)
            # Let's verify the standard LUW recurrence.
            # Lazebnik, Ustimenko, Woldar (1998):
            # y_1 = x_1
            # y_i = x_i + sum_{j=1}^{i-1} x_j y_j
            # Wait, is it x_j y_j or x_j y_{j+1}?
            # The graph is defined by equations:
            # y_1 = x_1
            # y_i = x_i + sum_{j < i} c_{i,j} x_j y_j
            # For the "standard" LUW graph with girth 2d+2, c_{i,j} = 1.
            
            y = [0] * (d + 1)
            y[1] = coords[1]
            for i in range(2, d + 1):
                s = coords[i]
                for j in range(1, i):
                    s = (s + mul_t[coords[j]][y[j]]) % q
                y[i] = s
            
            # Encode y to index
            y_idx = 0
            for i in range(1, d + 1):
                y_idx += y[i] * (q ** (i - 1))
            
            # Add edge
            # Side 0 idx, Side 1 idx + q^d
            edges.append((idx, y_idx + q ** d))
            
        return n_verts, edges
    