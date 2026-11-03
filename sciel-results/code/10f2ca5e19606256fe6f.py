import numpy as np

def get_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indices:
    U[i*2 + j, r]
    V[j*2 + k, r]
    W[k*2 + i, r]
    """
    # Strassen multipliers
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e+f)
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # A indices: 00=0, 01=1, 10=2, 11=3
    # B indices: 00=0, 01=1, 10=2, 11=3
    # C indices: 00=0, 10=1, 01=2, 11=3 (k*2+i)
    # Note: W index is k*2+i.
    # C_00 is i=0,k=0 -> idx 0
    # C_01 is i=0,k=1 -> idx 2
    # C_10 is i=1,k=0 -> idx 1
    # C_11 is i=1,k=1 -> idx 3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: m1
    # u1 = a+d -> U[0,0]=1, U[3,0]=1
    # v1 = e+h -> V[0,0]=1, V[3,0]=1
    # C_00 += m1, C_11 += m1
    # W[0,0]=1 (C_00), W[3,0]=1 (C_11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: m2
    # u2 = c+d -> U[2,1]=1, U[3,1]=1
    # v2 = e+f -> V[0,1]=1, V[1,1]=1
    # C_00 -= m2, C_10 += m2
    # W[0,1]=-1 (C_00), W[1,1]=1 (C_10)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1; V[1, 1] = 1
    W[0, 1] = -1; W[1, 1] = 1
    
    # r=2: m3
    # u3 = a -> U[0,2]=1
    # v3 = f-h -> V[1,2]=1, V[3,2]=-1
    # C_00 += m3, C_10 -= m3
    # W[0,2]=1 (C_00), W[1,2]=-1 (C_10)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = -1
    
    # r=3: m4
    # u4 = d -> U[3,3]=1
    # v4 = g-e -> V[2,3]=1, V[0,3]=-1
    # C_01 += m4, C_11 -= m4
    # W[2,3]=1 (C_01), W[3,3]=-1 (C_11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = -1
    
    # r=4: m5
    # u5 = a+b -> U[0,4]=1, U[1,4]=1
    # v5 = h -> V[3,4]=1
    # C_00 += m5, C_01 += m5
    # W[0,4]=1 (C_00), W[2,4]=1 (C_01)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # r=5: m6
    # u6 = c-a -> U[2,5]=1, U[0,5]=-1
    # v6 = e+f -> V[0,5]=1, V[1,5]=1
    # C_10 += m6
    # W[1,5]=1 (C_10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[1, 5] = 1
    
    # r=6: m7
    # u7 = b-d -> U[1,6]=1, U[3,6]=-1
    # v7 = g+h -> V[2,6]=1, V[3,6]=1
    # C_01 -= m7, C_11 += m7
    # W[2,6]=-1 (C_01), W[3,6]=1 (C_11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[2, 6] = -1; W[3, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    # Rank 108 decomposition
    # Structure:
    # 2 blocks of (2, 5, 6)
    # Each (2, 5, 6) is:
    #   2 blocks of (2, 2, 6)
    #   1 block of (2, 1, 6)
    # Each (2, 2, 6) is:
    #   3 blocks of (2, 2, 2) [Strassen]
    
    # We will build the full U, V, W matrices by assembling these blocks.
    # Total rank = 2 * (2 * (3 * 7) + 1 * (2 * 1 * 6)) = 2 * (42 + 12) = 2 * 54 = 108.
    
    rank = 108
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_str, V_str, W_str = get_strassen_2x2x2()
    
    # Helper to place a block
    def place_block(U_base, V_base, W_base, 
                    i_start, j_start, k_start, 
                    i_len, j_len, k_len, 
                    U_target, V_target, W_target, 
                    rank_offset):
        # Map local indices to global indices
        # Local i in [0, i_len), j in [0, j_len), k in [0, k_len)
        # Global i = i_start + i
        # Global j = j_start + j
        # Global k = k_start + k
        
        # U index: global_i * m + global_j
        # V index: global_j * p + global_k
        # W index: global_k * n + global_i
        
        for r in range(U_base.shape[1]):
            for i in range(i_len):
                for j in range(j_len):
                    u_val = U_base[i * j_len + j, r] # Wait, U_base is for specific dims
                    # We need to be careful with the shapes of U_base
                    # For Strassen, U_base is (4, 7) for 2x2.
                    # If we are placing a 2x2 block, i_len=2, j_len=2.
                    # U_base index is i*2 + j.
                    # This works if U_base was generated for that specific i_len, j_len.
                    
                    gi = i_start + i
                    gj = j_start + j
                    U_target[gi * m + gj, rank_offset + r] += u_val
                    
            for j in range(j_len):
                for k in range(k_len):
                    v_val = V_base[j * k_len + k, r]
                    gj = j_start + j
                    gk = k_start + k
                    V_target[gj * p + gk, rank_offset + r] += v_val
                    
            for k in range(k_len):
                for i in range(i_len):
                    w_val = W_base[k * i_len + i, r]
                    gk = k_start + k
                    gi = i_start + i
                    W_target[gk * n + gi, rank_offset + r] += w_val

    # We need base blocks for (2,2,2), (2,2,6), (2,1,6), (2,5,6), (4,5,6)
    # It's easier to just iterate through the logical structure and place the Strassen blocks directly,
    # or create intermediate factor matrices.
    # Given the complexity, let's just create the (2,2,6) and (2,1,6) factors first.
    
    # (2,2,6) factors
    # Composed of 3 (2,2,2) blocks.
    # j is 0,1. k is 0..5.
    # Block 0: k=0,1. Block 1: k=2,3. Block 2: k=4,5.
    U_226 = np.zeros((4, 21))
    V_226 = np.zeros((12, 21)) # 2*6
    W_226 = np.zeros((12, 21)) # 6*2
    
    for b in range(3):
        k_start = b * 2
        place_block(U_str, V_str, W_str, 
                    0, 0, k_start, 
                    2, 2, 2, 
                    U_226, V_226, W_226, 
                    b * 7)
                    
    # (2,1,6) factors
    # Naive: rank 12.
    # U (2*1=2, 12), V (1*6=6, 12), W (6*2=12, 12)
    U_216 = np.zeros((2, 12))
    V_216 = np.zeros((6, 12))
    W_216 = np.zeros((12, 12))
    
    col = 0
    for i in range(2):
        for k in range(6):
            U_216[i * 1 + 0, col] = 1
            V_216[0 * 6 + k, col] = 1
            W_216[k * 2 + i, col] = 1
            col += 1
            
    # (2,5,6) factors
    # Composed of two (2,2,6) and one (2,1,6).
    # j is 0..4.
    # Block 0: j=0,1. (2,2,6)
    # Block 1: j=2,3. (2,2,6)
    # Block 2: j=4.   (2,1,6)
    U_256 = np.zeros((10, 54)) # 2*5
    V_256 = np.zeros((30, 54)) # 5*6
    W_256 = np.zeros((12, 54)) # 6*2
    
    # Place Block 0 (j=0,1)
    place_block(U_226, V_226, W_226,
                0, 0, 0,
                2, 2, 6,
                U_256, V_256, W_256,
                0)
                
    # Place Block 1 (j=2,3)
    place_block(U_226, V_226, W_226,
                0, 2, 0,
                2, 2, 6,
                U_256, V_256, W_256,
                21)
                
    # Place Block 2 (j=4)
    place_block(U_216, V_216, W_216,
                0, 4, 0,
                2, 1, 6,
                U_256, V_256, W_256,
                42)
                
    # (4,5,6) factors
    # Composed of two (2,5,6).
    # i is 0..3.
    # Block 0: i=0,1.
    # Block 1: i=2,3.
    U_456 = np.zeros((20, 108)) # 4*5
    V_456 = np.zeros((30, 108)) # 5*6
    W_456 = np.zeros((24, 108)) # 6*4
    
    place_block(U_256, V_256, W_256,
                0, 0, 0,
                2, 5, 6,
                U_456, V_456, W_456,
                0)
                
    place_block(U_256, V_256, W_256,
                2, 0, 0,
                2, 5, 6,
                U_456, V_456, W_456,
                54)
                
    return {"U": U_456.tolist(), "V": V_456.tolist(), "W": W_456.tolist()}

def construct():
    n, m, p = 4, 5, 6
    # Rank 108 decomposition
    # Structure:
    # 2 blocks of (2, 5, 6)
    # Each (2, 5, 6) is:
    #   2 blocks of (2, 2, 6)
    #   1 block of (2, 1, 6)
    # Each (2, 2, 6) is:
    #   3 blocks of (2, 2, 2) [Strassen]
    
    # We will build the full U, V, W matrices by assembling these blocks.
    # Total rank = 2 * (2 * (3 * 7) + 1 * (2 * 1 * 6)) = 2 * (42 + 12) = 2 * 54 = 108.
    
    rank = 108
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_str, V_str, W_str = get_strassen_2x2x2()
    
    # Helper to place a block
    def place_block(U_base, V_base, W_base, 
                    i_start, j_start, k_start, 
                    i_len, j_len, k_len, 
                    U_target, V_target, W_target, 
                    rank_offset):
        # Map local indices to global indices
        # Local i in [0, i_len), j in [0, j_len), k in [0, k_len)
        # Global i = i_start + i
        # Global j = j_start + j
        # Global k = k_start + k
        
        # U index: global_i * m + global_j
        # V index: global_j * p + global_k
        # W index: global_k * n + global_i
        
        for r in range(U_base.shape[1]):
            for i in range(i_len):
                for j in range(j_len):
                    u_val = U_base[i * j_len + j, r]
                    gi = i_start + i
                    gj = j_start + j
                    U_target[gi * m + gj, rank_offset + r] += u_val
                    
            for j in range(j_len):
                for k in range(k_len):
                    v_val = V_base[j * k_len + k, r]
                    gj = j_start + j
                    gk = k_start + k
                    V_target[gj * p + gk, rank_offset + r] += v_val
                    
            for k in range(k_len):
                for i in range(i_len):
                    w_val = W_base[k * i_len + i, r]
                    gk = k_start + k
                    gi = i_start + i
                    W_target[gk * n + gi, rank_offset + r] += w_val

    # We need base blocks for (2,2,2), (2,2,6), (2,1,6), (2,5,6), (4,5,6)
    # It's easier to just iterate through the logical structure and place the Strassen blocks directly,
    # or create intermediate factor matrices.
    # Given the complexity, let's just create the (2,2,6) and (2,1,6) factors first.
    
    # (2,2,6) factors
    # Composed of 3 (2,2,2) blocks.
    # j is 0,1. k is 0..5.
    # Block 0: k=0,1. Block 1: k=2,3. Block 2: k=4,5.
    U_226 = np.zeros((4, 21))
    V_226 = np.zeros((12, 21)) # 2*6
    W_226 = np.zeros((12, 21)) # 6*2
    
    for b in range(3):
        k_start = b * 2
        place_block(U_str, V_str, W_str, 
                    0, 0, k_start, 
                    2, 2, 2, 
                    U_226, V_226, W_226, 
                    b * 7)
                    
    # (2,1,6) factors
    # Naive: rank 12.
    # U (2*1=2, 12), V (1*6=6, 12), W (6*2=12, 12)
    U_216 = np.zeros((2, 12))
    V_216 = np.zeros((6, 12))
    W_216 = np.zeros((12, 12))
    
    col = 0
    for i in range(2):
        for k in range(6):
            U_216[i * 1 + 0, col] = 1
            V_216[0 * 6 + k, col] = 1
            W_216[k * 2 + i, col] = 1
            col += 1
            
    # (2,5,6) factors
    # Composed of two (2,2,6) and one (2,1,6).
    # j is 0..4.
    # Block 0: j=0,1. (2,2,6)
    # Block 1: j=2,3. (2,2,6)
    # Block 2: j=4.   (2,1,6)
    U_256 = np.zeros((10, 54)) # 2*5
    V_256 = np.zeros((30, 54)) # 5*6
    W_256 = np.zeros((12, 54)) # 6*2
    
    # Place Block 0 (j=0,1)
    place_block(U_226, V_226, W_226,
                0, 0, 0,
                2, 2, 6,
                U_256, V_256, W_256,
                0)
                
    # Place Block 1 (j=2,3)
    place_block(U_226, V_226, W_226,
                0, 2, 0,
                2, 2, 6,
                U_256, V_256, W_256,
                21)
                
    # Place Block 2 (j=4)
    place_block(U_216, V_216, W_216,
                0, 4, 0,
                2, 1, 6,
                U_256, V_256, W_256,
                42)
                
    # (4,5,6) factors
    # Composed of two (2,5,6).
    # i is 0..3.
    # Block 0: i=0,1.
    # Block 1: i=2,3.
    U_456 = np.zeros((20, 108)) # 4*5
    V_456 = np.zeros((30, 108)) # 5*6
    W_456 = np.zeros((24, 108)) # 6*4
    
    place_block(U_256, V_256, W_256,
                0, 0, 0,
                2, 5, 6,
                U_456, V_456, W_456,
                0)
                
    place_block(U_256, V_256, W_256,
                2, 0, 0,
                2, 5, 6,
                U_456, V_456, W_456,
                54)
                
    return {"U": U_456.tolist(), "V": V_456.tolist(), "W": W_456.tolist()}
