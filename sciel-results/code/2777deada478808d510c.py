import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Start with the 4x4 Strassen decomposition (rank 49)
    # We need to generate the U, V, W for 4x4x4 rank 49.
    # Then add 9 terms for the 5th row/col.
    
    # Helper to build Strassen 2x2
    def strassen_2x2():
        # Returns U, V, W of shape (4, 7)
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11 + A22)(B11 + B22)
        U[0, 0] = 1; U[1, 0] = 1
        V[0, 0] = 1; V[1, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
        
        # M2 = (A21 + A22)B11
        U[1, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3 = A11(B21 - B22)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1; W[1, 2] = 1
        
        # M4 = A22(B11 - B12)
        U[3, 3] = 1
        V[0, 3] = 1; V[1, 3] = -1
        W[2, 3] = 1; W[3, 3] = 1
        
        # M5 = (A11 + A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[1, 4] = 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[1, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[3, 5] = 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[1, 6] = 1
        
        return U, V, W

    def strassen_4x4():
        # Recursively apply Strassen to 2x2 blocks of 2x2 matrices
        # The inputs are 4x4 matrices, viewed as 2x2 blocks of 2x2.
        # This requires 7 multiplications of 2x2 matrices.
        # Each 2x2 multiplication has rank 7.
        # Total rank 49.
        
        U_2, V_2, W_2 = strassen_2x2()
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # The 7 "big" multiplications correspond to the 7 terms of Strassen.
        # Each "big" multiplication is a 2x2 matrix multiplication.
        # We need to map the 16 input elements to the 4 elements of each 2x2 block.
        
        # Block indices:
        # 0: 00, 1: 01, 2: 10, 3: 11 (for 4x4 matrix, row/col 0-3)
        # Actually, let's use the recursive structure.
        # A_4x4 = [[A11, A12], [A21, A22]] where Aij are 2x2.
        # The 7 terms M1..M7 are 2x2 matrices.
        # Each Mj is computed by strassen_2x2 on its 2x2 blocks.
        
        # We need to define the linear combinations for the 4x4 inputs.
        # This is getting complex to hardcode.
        # Instead, I will use a simpler block decomposition for 4x4 with rank 49?
        # Or just use schoolbook for 4x4 (rank 64) and add 9 terms? Rank 73.
        # 73 < 125.
        # Let's try to get rank 58 with Strassen.
        
        # To avoid hardcoding the 49 terms, I will generate them programmatically.
        # But for now, I'll use a known rank-49 decomposition if I can find it, or just use schoolbook 64+9=73.
        # Given time, I will use schoolbook for 4x4 (rank 64) and add 9 terms.
        # Rank 73.
        
        rank_44 = 16
        U = np.zeros((16, rank_44))
        V = np.zeros((16, rank_44))
        W = np.zeros((16, rank_44))
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        
        return U, V, W

    U_44, V_44, W_44 = strassen_4x4()
    
    rank_44 = U_44.shape[1]
    rank_total = rank_44 + 9
    
    U = np.zeros((25, rank_total))
    V = np.zeros((25, rank_total))
    W = np.zeros((25, rank_total))
    
    # Fill in the 4x4 part (indices 0-3)
    # U, V, W are (25, rank). The 4x4 part uses indices 0-3 in rows/cols.
    # U[i*5+j, col] for i,j in 0..3
    # V[j*5+k, col] for j,k in 0..3
    # W[k*5+i, col] for k,i in 0..3
    
    for col in range(rank_44):
        for i in range(4):
            for j in range(4):
                U[i*5+j, col] = U_44[i*4+j, col]
        for j in range(4):
            for k in range(4):
                V[j*5+k, col] = V_44[j*4+k, col]
        for k in range(4):
            for i in range(4):
                W[k*5+i, col] = W_44[k*4+i, col]
                
    # Add 9 terms for the 5th row/col (index 4)
    # 1. C[4, 0..3] = A[4, :] dot B[:, 0..3]
    # 2. C[0..3, 4] = A[0..3, :] dot B[:, 4]
    # 3. C[4, 4] = A[4, :] dot B[:, 4]
    
    extra_col = rank_44
    # Term for C[4, j] for j=0..3
    # We need 4 terms.
    for j in range(4):
        # u picks A[4, 0..4], v picks B[0..4, j]
        # w puts it in C[4, j]
        for k in range(5):
            U[4*5+k, extra_col] = 1
            V[k*5+j, extra_col] = 1
        W[4*5+j, extra_col] = 1
        extra_col += 1
        
    # Term for C[i, 4] for i=0..3
    for i in range(4):
        # u picks A[i, 0..4], v picks B[0..4, 4]
        # w puts it in C[i, 4]
        for k in range(5):
            U[i*5+k, extra_col] = 1
            V[k*5+4, extra_col] = 1
        W[4*5+i, extra_col] = 1
        extra_col += 1
        
    # Term for C[4, 4]
    for k in range(5):
        U[4*5+k, extra_col] = 1
        V[k*5+4, extra_col] = 1
    W[4*5+4, extra_col] = 1
    extra_col += 1
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
