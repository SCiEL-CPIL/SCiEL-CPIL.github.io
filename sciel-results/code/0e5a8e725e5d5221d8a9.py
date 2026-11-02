import numpy as np

def _strassen_2x2_2x2_2x2():
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (b+d)(e+g)
    U[1, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[2, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)h
    U[2, 1] = 1; U[3, 1] = 1
    V[3, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = e(a+b)
    U[0, 3] = 1; U[1, 3] = 1
    V[0, 3] = 1
    W[0, 3] = -1; W[1, 3] = 1
    
    # M5 = d(g-f)
    U[3, 4] = 1
    V[2, 4] = 1; V[1, 4] = -1
    W[2, 4] = 1; W[3, 4] = 1
    
    # M6 = (a+c)(h+g)
    U[0, 5] = 1; U[2, 5] = 1
    V[3, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1
    
    # M7 = (b+c)(e+f)
    U[1, 6] = 1; U[2, 6] = 1
    V[0, 6] = 1; V[1, 6] = 1
    W[0, 6] = -1
    
    return U, V, W

def _build_4x4_4x4_strassen():
    U2, V2, W2 = _strassen_2x2_2x2_2x2()
    
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    # Helper to get global indices for a 2x2 block in a 4x4 matrix
    def get_indices(block_r, block_c):
        return [
            (2*block_r)*4 + (2*block_c),
            (2*block_r)*4 + (2*block_c+1),
            (2*block_r+1)*4 + (2*block_c),
            (2*block_r+1)*4 + (2*block_c+1)
        ]
    
    # Strassen steps
    # M1 = (A01+A11)(B00+B10) -> C00+=M1, C11+=M1
    # M2 = (A10+A11)B11 -> C10+=M2, C11-=M2
    # M3 = A00(B01-B11) -> C01+=M3, C11+=M3
    # M4 = B00(A00+A01) -> C00-=M4, C01+=M4
    # M5 = A11(B10-B01) -> C10+=M5, C11+=M5
    # M6 = (A00+A10)(B11+B10) -> C00+=M6
    # M7 = (A01+A10)(B00+B01) -> C00-=M7
    
    # For each M, we add the rank-7 decomposition.
    # The "A" part of M is a sum of blocks.
    # So U_M = sum(U_block for blocks in A_part)
    # V_M = sum(V_block for blocks in B_part)
    # W_M = sum(W_block for blocks in C_part)
    
    # We iterate over the 7 sub-tensors of the rank-7 decomposition.
    for sub in range(7):
        col = sub # This is the column within the 7 for this M? No, we need to map to global columns.
        # Actually, for each M, we use 7 columns.
        # Let's just fill the 49 columns.
        
        # This loop structure is better:
        # for m in range(7):
        #   for sub in range(7):
        #      col = m*7 + sub
        #      U[:, col] = ...
        
    for m in range(7):
        # Define blocks for A, B, C for this M
        # m=0: M1
        if m == 0:
            a_blocks = [(0,1), (1,1)] # 01, 11
            b_blocks = [(0,0), (1,0)] # 00, 10
            c_blocks = [(0,0), (1,1)] # 00, 11
            c_signs = [1, 1]
        elif m == 1:
            a_blocks = [(1,0), (1,1)]
            b_blocks = [(1,1)]
            c_blocks = [(1,0), (1,1)]
            c_signs = [1, -1]
        elif m == 2:
            a_blocks = [(0,0)]
            b_blocks = [(0,1), (1,1)]
            c_blocks = [(0,1), (1,1)]
            c_signs = [1, 1]
        elif m == 3:
            a_blocks = [(0,0), (0,1)]
            b_blocks = [(0,0)]
            c_blocks = [(0,0), (0,1)]
            c_signs = [-1, 1]
        elif m == 4:
            a_blocks = [(1,1)]
            b_blocks = [(1,0), (0,1)]
            c_blocks = [(1,0), (1,1)]
            c_signs = [1, 1]
        elif m == 5:
            a_blocks = [(0,0), (1,0)]
            b_blocks = [(1,1), (1,0)]
            c_blocks = [(0,0)]
            c_signs = [1]
        elif m == 6:
            a_blocks = [(0,1), (1,0)]
            b_blocks = [(0,0), (0,1)]
            c_blocks = [(0,0)]
            c_signs = [-1]
            
        for sub in range(7):
            col = m*7 + sub
            
            # Sum U vectors for A blocks
            u_vec = np.zeros(16)
            for br, bc in a_blocks:
                idx = get_indices(br, bc)
                u_vec[idx] += U2[:, sub]
            
            # Sum V vectors for B blocks
            v_vec = np.zeros(16)
            for br, bc in b_blocks:
                idx = get_indices(br, bc)
                v_vec[idx] += V2[:, sub]
            
            # Sum W vectors for C blocks
            w_vec = np.zeros(16)
            for i, (br, bc) in enumerate(c_blocks):
                idx = get_indices(br, bc)
                w_vec[idx] += c_signs[i] * W2[:, sub]
            
            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            
    return U, V, W
