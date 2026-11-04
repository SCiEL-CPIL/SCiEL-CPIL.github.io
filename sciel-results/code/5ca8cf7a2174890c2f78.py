import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Sub-tensor decomposition: split into two 2x3 x 3x3 problems
    # For each half (c in {0,1}), use rank-9 decomposition of 2x3x3 tensor
    
    # Rank-9 decomposition of 2x3x3 matrix multiplication tensor
    # Using the known optimal decomposition for 2x3x3
    # The 2x3x3 tensor has rank 9
    
    def add_2x3x3_decomposition(U, V, V_offset, W_offset, col_start):
        """Add rank-9 decomposition for 2x3 x 3x3 multiplication.
        
        U indices: 0-5 (2x3 matrix A, row-major)
        V indices: 0-8 (3x3 matrix B, row-major, offset by V_offset)
        W indices: 0-5 (2x3 output C, row-major, offset by W_offset)
        """
        col = col_start
        
        # Rank-9 decomposition for 2x3x3:
        # C[i,j] = sum_k A[i,k] * B[k,j]
        # We need 9 terms to decompose this
        
        # Using the structure: for each output position (i,j),
        # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] + A[i,2]*B[2,j]
        # That's 6*3=18 terms for schoolbook, but we can do better.
        
        # Known rank-9 decomposition for 2x3x3:
        # Let's use the following scheme based on Strassen-like ideas
        
        # Actually, let me use a verified rank-9 decomposition.
        # For 2x3 x 3x3, the rank is known to be 9.
        
        # Decomposition terms:
        # Term 0: A[0,0] * B[0,0] -> C[0,0]
        U[0][col] = 1
        V[V_offset + 0][col] = 1
        W[W_offset + 0][col] = 1
        col += 1
        
        # Term 1: A[0,0] * B[0,1] -> C[0,1]
        U[0][col] = 1
        V[V_offset + 1][col] = 1
        W[W_offset + 1][col] = 1
        col += 1
        
        # Term 2: A[0,0] * B[0,2] -> C[0,2]
        U[0][col] = 1
        V[V_offset + 2][col] = 1
        W[W_offset + 2][col] = 1
        col += 1
        
        # Term 3: A[1,0] * B[0,0] -> C[1,0]
        U[3][col] = 1
        V[V_offset + 0][col] = 1
        W[W_offset + 3][col] = 1
        col += 1
        
        # Term 4: A[1,0] * B[0,1] -> C[1,1]
        U[3][col] = 1
        V[V_offset + 1][col] = 1
        W[W_offset + 4][col] = 1
        col += 1
        
        # Term 5: A[1,0] * B[0,2] -> C[1,2]
        U[3][col] = 1
        V[V_offset + 2][col] = 1
        W[W_offset + 5][col] = 1
        col += 1
        
        # Term 6: A[0,1] * B[1,0] + A[1,1] * B[0,0] -> combined
        # This approach isn't working simply. Let me use a different strategy.
        # 
        # Actually, the rank of the 2x3x3 tensor is NOT 9 in general.
        # Let me reconsider. The tensor rank of [2,3,3] matrix multiplication
        # is at most 18 (schoolbook) and at least... 
        # 
        # Wait, I need to reconsider the problem structure.
        # The full tensor is T[i,j,k] where i in [0,5] (2x3 A), j in [0,17] (3x6 B), k in [0,11] (2x6 C)
        # T[i,j,k] = 1 if the multiplication A*B produces output k from inputs i,j
        #
        # For C[r,c] = sum_s A[r,s] * B[s,c]
        # In our indexing: A index = r*m+s, B index = s*p+c, C index = c*n+r
        #
        # So T[r*m+s, s*p+c, c*n+r] = 1
        #
        # This tensor factors as: for each (r,c), T involves A[r,*] and B[*,c]
        # The sub-tensor for fixed c has rank at most 2*3 = 6 (one term per (r,s) pair)
        # But we can share terms across different c values.
        
        # Let me just use the schoolbook for each half: 2*3*3 = 18 per half, total 36.
        # That's worse than 33. So I need a better approach.
        
        # Let me go back to the original approach but verify it's correct.
        return col
    
    # The sub-tensor idea as stated doesn't immediately give rank 18.
    # Let me keep the original rank-33 solution but verify it carefully.
    
    # Reset and use the original approach
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_1 (col 0) * B_1 (row 0): 2x1x6 schoolbook, rank 12
    for k in range(p):
        U[0][col] = 1
        V[k][col] = 1
        W[2*k][col] = 1
        col += 1
        U[3][col] = 1
        V[k][col] = 1
        W[2*k + 1][col] = 1
        col += 1
    
    # Part 2: A_2 (cols 1,2) * B_2 (rows 1,2): 2x2x6 via 3 Strassen 2x2x2 blocks
    for x in range(3):
        cs = 2 * x
        # M1: (a00+a11)*(b00+b11)
        U[1][col] = 1; U[5][col] = 1
        V[6 + cs][col] = 1; V[12 + cs + 1][col] = 1
        W[2*cs][col] = 1; W[2*cs + 3][col] = 1
        col += 1
        # M2: (a10+a11)*b00
        U[4][col] = 1; U[5][col] = 1
        V[6 + cs][col] = 1
        W[2*cs + 1][col] = 1; W[2*cs + 3][col] = -1
        col += 1
        # M3: a00*(b01-b11)
        U[1][col] = 1
        V[6 + cs + 1][col] = 1; V[12 + cs + 1][col] = -1
        W[2*cs + 2][col] = 1; W[2*cs + 3][col] = 1
        col += 1
        # M4: a11*(b10-b00)
        U[5][col] = 1
        V[12 + cs][col] = 1; V[6 + cs][col] = -1
        W[2*cs][col] = 1; W[2*cs + 1][col] = 1
        col += 1
        # M5: (a00+a01)*b11
        U[1][col] = 1; U[2][col] = 1
        V[12 + cs + 1][col] = 1
        W[2*cs][col] = -1; W[2*cs + 2][col] = 1
        col += 1
        # M6: (a10-a00)*(b00+b01)
        U[4][col] = 1; U[1][col] = -1
        V[6 + cs][col] = 1; V[6 + cs + 1][col] = 1
        W[2*cs + 3][col] = 1
        col += 1
        # M7: (a01-a11)*(b10+b11)
        U[2][col] = 1; U[5][col] = -1
        V[12 + cs][col] = 1; V[12 + cs + 1][col] = 1
        W[2*cs][col] = 1
        col += 1
    
    # Quick verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T_est = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    T_target = np.zeros((n*m, m*p, p*n))
    for r in range(n):
        for s in range(m):
            for c in range(p):
                T_target[r*m+s, s*p+c, c*n+r] = 1
    if np.max(np.abs(T_est - T_target)) > 1e-9:
        raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
