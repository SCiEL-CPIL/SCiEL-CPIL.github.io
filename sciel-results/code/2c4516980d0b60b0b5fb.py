import numpy as np

def construct():
    n, m, p = 3, 4, 6
    
    # We build the decomposition recursively.
    # Each step returns U, V, W for a sub-problem.
    # We need to map local indices to global indices.
    
    def make_strassen(n=2, m=2, p=2):
        # Returns U, V, W for 2x2x2
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        # Indices:
        # A: 0,1,2,3 (a,b,c,d)
        # B: 0,1,2,3 (e,f,g,h)
        # C: 0,1,2,3 (C00, C10, C01, C11) -> k*n+i
        # C00: k=0,i=0 -> 0
        # C10: k=0,i=1 -> 1
        # C01: k=1,i=0 -> 2
        # C11: k=1,i=1 -> 3
        
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # M1 = (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
        
        # M2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = 1
        
        # M3 = a(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # M4 = d(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[3, 4] = -1
        
        # M6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W

    def make_fft_221(n=2, m=2, p=1):
        # Returns U, V, W for 2x2x1
        # U shape (4, 4), V shape (2, 4), W shape (2, 4)
        U = np.zeros((4, 4), dtype=float)
        V = np.zeros((2, 4), dtype=float)
        W = np.zeros((2, 4), dtype=float)
        
        # M1 = (a+b)(e+f)
        U[0, 0] = 1; U[1, 0] = 1
        V[0, 0] = 1; V[1, 0] = 1
        W[0, 0] = 0.5; W[1, 0] = 0.5
        
        # M2 = (a-b)(e-f)
        U[0, 1] = 1; U[1, 1] = -1
        V[0, 1] = 1; V[1, 1] = -1
        W[0, 1] = 0.5; W[1, 1] = 0.5
        
        # M3 = (c+d)(e+f)
        U[2, 2] = 1; U[3, 2] = 1
        V[0, 2] = 1; V[1, 2] = 1
        W[0, 2] = 0.5; W[1, 2] = 0.5
        
        # M4 = (c-d)(e-f)
        U[2, 3] = 1; U[3, 3] = -1
        V[0, 3] = 1; V[1, 3] = -1
        W[0, 3] = 0.5; W[1, 3] = 0.5
        
        return U, V, W

    def make_fft_123(n=1, m=2, p=3):
        # Returns U, V, W for 1x2x3
        # U shape (2, 6), V shape (6, 6), W shape (3, 6)
        U = np.zeros((2, 6), dtype=float)
        V = np.zeros((6, 6), dtype=float)
        W = np.zeros((3, 6), dtype=float)
        
        # C_k = (a+b)S1_k + (a-b)S2_k all over 2
        # S1 = [e1+e2, f1+f2, g1+g2]
        # S2 = [e1-e2, f1-f2, g1-g2]
        
        # Col 0: (a+b)(e1+e2)
        U[0, 0] = 1; U[1, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1 # e1, e2
        W[0, 0] = 0.5
        
        # Col 1: (a-b)(e1-e2)
        U[0, 1] = 1; U[1, 1] = -1
        V[0, 1] = 1; V[3, 1] = -1
        W[0, 1] = 0.5
        
        # Col 2: (a+b)(f1+f2)
        U[0, 2] = 1; U[1, 2] = 1
        V[1, 2] = 1; V[4, 2] = 1 # f1, f2
        W[1, 2] = 0.5
        
        # Col 3: (a-b)(f1-f2)
        U[0, 3] = 1; U[1, 3] = -1
        V[1, 3] = 1; V[4, 3] = -1
        W[1, 3] = 0.5
        
        # Col 4: (a+b)(g1+g2)
        U[0, 4] = 1; U[1, 4] = 1
        V[2, 4] = 1; V[5, 4] = 1 # g1, g2
        W[2, 4] = 0.5
        
        # Col 5: (a-b)(g1-g2)
        U[0, 5] = 1; U[1, 5] = -1
        V[2, 5] = 1; V[5, 5] = -1
        W[2, 5] = 0.5
        
        return U, V, W

    def combine_decomps(decomp1, decomp2, n1, m1, p1, n2, m2, p2, offset_u, offset_v, offset_w):
        # This is complex to do generically. 
        # Instead, I will build the full 3x4x6 matrix directly by placing blocks.
        pass

    # Let's build the full U, V, W for 3x4x6.
    # Total Rank = 68.
    rank = 68
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    # Part 1: A_left (3x2) * B_top (2x6)
    # A_left uses A cols 0,1. Indices 0,1, 3,4, 6,7 in U (3x4 -> 12 entries).
    # B_top uses B rows 0,1. Indices 0..5 in V (4x6 -> 24 entries).
    # Output C (3x6).
    
    # We need to map the sub-decomposition indices to the global indices.
    
    # Helper to add a sub-decomposition
    def add_sub(U_sub, V_sub, W_sub, u_map, v_map, w_map, start_col):
        nonlocal col
        for r in range(U_sub.shape[1]):
            for i, ui in enumerate(U_sub[:, r]):
                if ui != 0:
                    U[u_map[i], start_col + r] += ui
            for i, vi in enumerate(V_sub[:, r]):
                if vi != 0:
                    V[v_map[i], start_col + r] += vi
            for i, wi in enumerate(W_sub[:, r]):
                if wi != 0:
                    W[w_map[i], start_col + r] += wi
        col += U_sub.shape[1]

    # We need to construct the 3x2x6 decomposition first.
    # 3x2x6 is sum of two 3x2x3.
    # 3x2x3 is sum of 2x2x3 (rows 0,1) and 1x2x3 (row 2).
    # 2x2x3 is sum of 2x2x2 (cols 0,1) and 2x2x1 (col 2).
    
    # This manual assembly is error-prone. 
    # I will write a function that generates the U, V, W for 3x2x6 
    # with local indices, then map to global.
    
    def get_323_decomp():
        # Returns U (6, 17), V (6, 17), W (9, 17) for 3x2x3
        # n=3, m=2, p=3
        # U indices: 0..5 (A is 3x2)
        # V indices: 0..5 (B is 2x3)
        # W indices: 0..8 (C is 3x3)
        
        U = np.zeros((6, 17), dtype=float)
        V = np.zeros((6, 17), dtype=float)
        W = np.zeros((9, 17), dtype=float)
        
        col = 0
        # Part 1: 2x2x3 (Rows 0,1 of A)
        # A_sub: rows 0,1 -> indices 0,1, 2,3 in U
        # B_sub: all cols -> indices 0..5 in V
        # C_sub: rows 0,1 -> indices 0..5 in W (k*n+i, n=3)
        # C00->0, C10->1, C01->3, C11->4, C02->6, C12->7
        
        # Get 2x2x3 decomp
        U_223, V_223, W_223 = get_223_decomp()
        # Map U_223 (4, 11) to U (6, 17)
        # U_223 indices 0,1,2,3 -> U indices 0,1,2,3
        U[0:4, col:col+11] = U_223
        V[0:6, col:col+11] = V_223 # V_223 is (6, 11)? No, 2x2x3 V is 2x3=6 entries.
        # Wait, V for 2x2x3 is 2x3 matrix, so 6 entries.
        # W_223 is 2x3 matrix, so 6 entries.
        # W indices:
        # C00 (0,0) -> 0
        # C10 (1,0) -> 1
        # C01 (0,1) -> 3
        # C11 (1,1) -> 4
        # C02 (0,2) -> 6
        # C12 (1,2) -> 7
        w_map_223 = [0, 1, 3, 4, 6, 7]
        for i in range(6):
            W[w_map_223[i], col:col+11] = W_223[i, :]
        col += 11
        
        # Part 2: 1x2x3 (Row 2 of A)
        # A_sub: row 2 -> index 4,5 in U
        # B_sub: all cols -> indices 0..5 in V
        # C_sub: row 2 -> indices 2, 5, 8 in W
        # C20 (2,0) -> 2
        # C21 (2,1) -> 5
        # C22 (2,2) -> 8
        
        U_123, V_123, W_123 = get_123_decomp()
        # U_123 is (2, 6). Map to U[4:6, col:col+6]
        U[4:6, col:col+6] = U_123
        V[0:6, col:col+6] = V_123
        w_map_123 = [2, 5, 8]
        for i in range(3):
            W[w_map_123[i], col:col+6] = W_123[i, :]
        col += 6
        
        return U, V, W

    def get_223_decomp():
        # Returns U (4, 11), V (6, 11), W (6, 11) for 2x2x3
        # n=2, m=2, p=3
        U = np.zeros((4, 11), dtype=float)
        V = np.zeros((6, 11), dtype=float)
        W = np.zeros((6, 11), dtype=float)
        
        col = 0
        # Part 1: 2x2x2 (Cols 0,1)
        U_222, V_222, W_222 = make_strassen()
        U[0:4, col:col+7] = U_222
        # V_222 is (4, 7). Map to V (6, 11).
        # V indices for 2x2: 0,1,2,3.
        # V indices for 2x3:
        # B is 2x3.
        # Col 0: 0,1
        # Col 1: 2,3
        # Col 2: 4,5
        # So V_222 indices 0,1,2,3 map to V indices 0,1,2,3.
        V[0:4, col:col+7] = V_222
        # W_222 is (4, 7). Map to W (6, 11).
        # W indices for 2x2: 0,1,2,3.
        # W indices for 2x3:
        # C is 2x3.
        # k=0: 0,1
        # k=1: 2,3
        # k=2: 4,5
        # So W_222 indices 0,1,2,3 map to W indices 0,1,2,3.
        W[0:4, col:col+7] = W_222
        col += 7
        
        # Part 2: 2x2x1 (Col 2)
        U_221, V_221, W_221 = make_fft_221()
        U[0:4, col:col+4] = U_221
        # V_221 is (2, 4). Map to V (6, 11).
        # V indices for 2x1: 0,1.
        # In 2x3, Col 2 is indices 4,5.
        V[4:6, col:col+4] = V_221
        # W_221 is (2, 4). Map to W (6, 11).
        # W indices for 2x1: 0,1.
        # In 2x3, Col 2 is indices 4,5.
        W[4:6, col:col+4] = W_221
        col += 4
        
        return U, V, W

    def get_123_decomp():
        # Returns U (2, 6), V (6, 6), W (3, 6) for 1x2x3
        return make_fft_123()

    def get_326_decomp():
        # Returns U (6, 34), V (12, 34), W (18, 34) for 3x2x6
        # n=3, m=2, p=6
        U = np.zeros((6, 34), dtype=float)
        V = np.zeros((12, 34), dtype=float)
        W = np.zeros((18, 34), dtype=float)
        
        col = 0
        # Part 1: 3x2x3 (Cols 0,1,2)
        U_323, V_323, W_323 = get_323_decomp()
        U[0:6, col:col+17] = U_323
        # V_323 is (6, 17). Map to V (12, 34).
        # V indices for 2x3: 0..5.
        # In 2x6, Cols 0,1,2 are indices 0..5.
        V[0:6, col:col+17] = V_323
        # W_323 is (9, 17). Map to W (18, 34).
        # W indices for 3x3: 0..8.
        # In 3x6, Cols 0,1,2:
        # k=0: 0,1,2
        # k=1: 3,4,5
        # k=2: 6,7,8
        # So W_323 indices 0..8 map to W indices 0..8.
        W[0:9, col:col+17] = W_323
        col += 17
        
        # Part 2: 3x2x3 (Cols 3,4,5)
        U_323_2, V_323_2, W_323_2 = get_323_decomp()
        U[0:6, col:col+17] = U_323_2
        # V indices for 2x3: 0..5.
        # In 2x6, Cols 3,4,5 are indices 6..11.
        V[6:12, col:col+17] = V_323_2
        # W indices for 3x3: 0..8.
        # In 3x6, Cols 3,4,5:
        # k=3: 9,10,11
        # k=4: 12,13,14
        # k=5: 15,16,17
        # So W_323 indices 0..8 map to W indices 9..17.
        W[9:18, col:col+17] = W_323_2
        col += 17
        
        return U, V, W

    # Now build 3x4x6
    # Part 1: A_left (3x2) * B_top (2x6)
    # A_left uses A cols 0,1. U indices 0,1, 3,4, 6,7.
    # B_top uses B rows 0,1. V indices 0..11.
    # Output C (3x6).
    
    U_326, V_326, W_326 = get_326_decomp()
    
    # Map U_326 (6, 34) to U (12, 68)
    # U indices:
    # A is 3x4.
    # Col 0: 0,1
    # Col 1: 2,3
    # Col 2: 4,5
    # Col 3: 6,7
    # Wait, A is 3x4.
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    # Row 2: 8,9,10,11
    # A_left (cols 0,1):
    # Row 0: 0,1
    # Row 1: 4,5
    # Row 2: 8,9
    # So U_326 indices 0,1,2,3,4,5 map to U indices 0,1,4,5,8,9.
    u_map = [0, 1, 4, 5, 8, 9]
    for i in range(6):
        U[u_map[i], 0:34] = U_326[i, :]
        
    # Map V_326 (12, 34) to V (24, 68)
    # B is 4x6.
    # Row 0: 0..5
    # Row 1: 6..11
    # Row 2: 12..17
    # Row 3: 18..23
    # B_top (rows 0,1):
    # Row 0: 0..5
    # Row 1: 6..11
    # So V_326 indices 0..11 map to V indices 0..11.
    V[0:12, 0:34] = V_326
    
    # Map W_326 (18, 34) to W (18, 68)
    # C is 3x6.
    # W indices match directly.
    W[0:18, 0:34] = W_326
    
    # Part 2: A_right (3x2) * B_bottom (2x6)
    # A_right uses A cols 2,3.
    # Row 0: 2,3
    # Row 1: 6,7
    # Row 2: 10,11
    # So U_326 indices 0..5 map to U indices 2,3,6,7,10,11.
    u_map2 = [2, 3, 6, 7, 10, 11]
    for i in range(6):
        U[u_map2[i], 34:68] = U_326[i, :]
        
    # B_bottom uses B rows 2,3.
    # Row 2: 12..17
    # Row 3: 18..23
    # So V_326 indices 0..11 map to V indices 12..23.
    V[12:24, 34:68] = V_326
    
    # W is same mapping.
    W[0:18, 34:68] = W_326
    
    return {"U": U, "V": V, "W": W}
