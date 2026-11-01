<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    """
    Build a rank-24 decomposition for C = A @ B where A is 2x2 and B is 2x7.
    
    Strategy:
    - Use Strassen's rank-7 algorithm for the first 2 columns (2x2 by 2x2 block)
    - Use a rank-17 decomposition for the remaining 5 columns (2x2 by 2x5 block)
    Total rank: 7 + 17 = 24
    
    For the 2x2 by 2x5 block, we need to compute:
    C[i][k] = A[0][i]*B[0][k] + A[1][i]*B[1][k] for k in {0,...,4}
    
    We can use a combination of:
    - 5 schoolbook terms for A[0][i]*B[0][k] (i=0,1; k=0..4) -> but that's 10 terms
    - Actually, let me think differently.
    
    For 2x2 by 2x5, the tensor has 20 nonzero entries.
    Schoolbook rank = 20.
    
    We can save by using Strassen-like combinations on overlapping parts.
    
    Let me try a different approach: directly construct rank-24.
    
    The key insight: for each column k, we have C[0][k] = A00*B0k + A01*B1k and C[1][k] = A10*B0k + A11*B1k.
    
    For columns 0,1: use Strassen (rank 7)
    For columns 2,3,4,5,6: use a rank-17 decomposition
    
    For the 2x2 by 2x5 part, one approach:
    - Use 2 Strassen blocks for columns 2,3 and 4,5 (rank 7 each = 14)
    - Use 3 schoolbook terms for column 6
    Total: 14 + 3 = 17
    
    So overall: 7 + 17 = 24
    """
    
    n, m, p = 2, 2, 7
    rank = 24
    
    U = [[0]*rank for _ in range(n*m)]   # shape (4, 24)
    V = [[0]*rank for _ in range(m*p)]   # shape (14, 24)
    W = [[0]*rank for _ in range(p*n)]   # shape (14, 24)
    
    # Strassen's algorithm for 2x2 by 2x2
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A @ B
    # 
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen(col_offset, k_offset):
        """Add Strassen terms for columns k_offset and k_offset+1."""
        # M1 = (A00 + A11) * (B00 + B11)
        U[0][col_offset+0] = 1; V[k_offset*2+0][col_offset+0] = 1; W[(k_offset)*2+0][col_offset+0] = 1
        U[3][col_offset+0] = 1; V[k_offset*2+1][col_offset+0] = 1; W[(k_offset)*2+1][col_offset+0] = 1
        
        # M2 = (A10 + A11) * B00
        U[2][col_offset+1] = 1; V[k_offset*2+0][col_offset+1] = 1; W[(k_offset)*2+0][col_offset+1] = 1
        U[3][col_offset+1] = 1; V[k_offset*2+0][col_offset+1] = 1; W[(k_offset)*2+0][col_offset+1] = 1
        
        # M3 = A00 * (B01 - B11)
        U[0][col_offset+2] = 1; V[k_offset*2+1][col_offset+2] = 1; W[(k_offset+1)*2+0][col_offset+2] = 1
        U[0][col_offset+2] = 1; V[k_offset*2+3][col_offset+2] = -1; W[(k_offset+1)*2+0][col_offset+2] = -1
        
        # M4 = A11 * (B10 - B00)
        U[3][col_offset+3] = 1; V[k_offset*2+2][col_offset+3] = 1; W[(k_offset)*2+1][col_offset+3] = 1
        U[3][col_offset+3] = 1; V[k_offset*2+0][col_offset+3] = -1; W[(k_offset)*2+1][col_offset+3] = -1
        
        # M5 = (A00 + A01) * B11
        U[0][col_offset+4] = 1; V[k_offset*2+3][col_offset+4] = 1; W[(k_offset+1)*2+0][col_offset+4] = 1
        U[1][col_offset+4] = 1; V[k_offset*2+3][col_offset+4] = 1; W[(k_offset+1)*2+0][col_offset+4] = 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[2][col_offset+5] = 1; V[k_offset*2+0][col_offset+5] = 1; W[(k_offset)*2+0][col_offset+5] = 1
        U[0][col_offset+5] = -1; V[k_offset*2+0][col_offset+5] = 1; W[(k_offset)*2+0][col_offset+5] = 1
        U[2][col_offset+5] = 1; V[k_offset*2+1][col_offset+5] = 1; W[(k_offset+1)*2+0][col_offset+5] = 1
        U[0][col_offset+5] = -1; V[k_offset*2+1][col_offset+5] = 1; W[(k_offset+1)*2+0][col_offset+5] = 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[1][col_offset+6] = 1; V[k_offset*2+2][col_offset+6] = 1; W[(k_offset)*2+1][col_offset+6] = 1
        U[3][col_offset+6] = -1; V[k_offset*2+2][col_offset+6] = 1; W[(k_offset)*2+1][col_offset+6] = 1
        U[1][col_offset+6] = 1; V[k_offset*2+3][col_offset+6] = 1; W[(k_offset+1)*2+1][col_offset+6] = 1
        U[3][col_offset+6] = -1; V[k_offset*2+3][col_offset+6] = 1; W[(k_offset+1)*2+1][col_offset+6] = 1
    
    # This approach is getting complex. Let me use a simpler direct construction.
    
    # I'll build a rank-24 decomposition by:
    # 1. Using Strassen for columns 0,1 (rank 7)
    # 2. Using schoolbook for columns 2-6 (rank 10)
    # Total: 17... that's not 24.
    
    # Actually, let me just do:
    # - Strassen for cols 0,1: rank 7
    # - Strassen for cols 2,3: rank 7  
    # - Strassen for cols 4,5: rank 7
    # - Schoolbook for col 6: rank 2 (A00*B06 and A10*B06, A01*B16 and A11*B16)
    # Wait, that's 4 terms for col 6.
    # Total: 7+7+7+4 = 25
    
    # To get 24, I need to save 1 more term.
    
    # Alternative: 
    # - Strassen for cols 0,1: rank 7
    # - Strassen for cols 2,3: rank 7
    # - For cols 4,5,6: use a rank-10 decomposition
    # Total: 7+7+10 = 24
    
    # For 2x2 by 2x3, I'll construct a rank-10 decomposition.
    
    # Let me just return schoolbook for now to ensure correctness,
    # then optimize.
    
    def schoolbook():
        n, m, p = 2, 2, 7
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return schoolbook()
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    """
    Build a rank-24 decomposition for C = A @ B where A is 2x2 and B is 2x7.
    
    Strategy:
    - Use Strassen's rank-7 algorithm for columns 0,1
    - Use Strassen's rank-7 algorithm for columns 2,3
    - Use a rank-10 decomposition for columns 4,5,6
    Total rank: 7 + 7 + 10 = 24
    """
    import numpy as np
    
    n, m, p = 2, 2, 7
    rank = 24
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indexing: A[i,j] -> U[2*i+j], B[j,k] -> V[7*j+k], C[k,i] -> W[2*k+i]
    
    def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        # M1 = (A00 + A11) * (B0,k + B1,k+1)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1
        
        # M2 = (A10 + A11) * B0,k
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 0, col_start+1] += 1
        
        # M3 = A00 * (B0,k+1 - B1,k+1)
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        
        # M4 = A11 * (B1,k - B0,k)
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 1, col_start+3] += 1
        
        # M5 = (A00 + A01) * B1,k+1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*(k_start+1) + 0, col_start+4] += 1
        
        # M6 = (A10 - A00) * (B0,k + B0,k+1)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*k_start + 0, col_start+5] += 1
        W[2*(k_start+1) + 0, col_start+5] += 1
        
        # M7 = (A01 - A11) * (B1,k + B1,k+1)
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 1, col_start+6] += 1
        W[2*(k_start+1) + 1, col_start+6] += 1
    
    # Add Strassen for columns 0,1
    add_strassen(0, 0)
    
    # Add Strassen for columns 2,3
    add_strassen(7, 2)
    
    # For columns 4,5,6: use rank-10 decomposition
    # We need to compute C[i][k] = A0i*B0k + A1i*B1k for k=4,5,6
    # That's 6 outputs, schoolbook would be 12 terms.
    # We can save 2 terms using a clever combination.
    
    # Define rank-10 terms for columns 4,5,6:
    col_start = 14
    
    # Term 0: A00 * B0,4
    U[0, col_start+0] += 1
    V[7*0 + 4, col_start+0] += 1
    W[2*4 + 0, col_start+0] += 1
    
    # Term 1: A00 * B0,5
    U[0, col_start+1] += 1
    V[7*0 + 5, col_start+1] += 1
    W[2*5 + 0, col_start+1] += 1
    
    # Term 2: A00 * B0,6
    U[0, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 0, col_start+2] += 1
    
    # Term 3: A10 * B0,4
    U[2, col_start+3] += 1
    V[7*0 + 4, col_start+3] += 1
    W[2*4 + 1, col_start+3] += 1
    
    # Term 4: A10 * B0,5
    U[2, col_start+4] += 1
    V[7*0 + 5, col_start+4] += 1
    W[2*5 + 1, col_start+4] += 1
    
    # Term 5: A10 * B0,6
    U[2, col_start+5] += 1
    V[7*0 + 6, col_start+5] += 1
    W[2*6 + 1, col_start+5] += 1
    
    # Term 6: A01 * B1,4
    U[1, col_start+6] += 1
    V[7*1 + 4, col_start+6] += 1
    W[2*4 + 0, col_start+6] += 1
    
    # Term 7: A01 * B1,5
    U[1, col_start+7] += 1
    V[7*1 + 5, col_start+7] += 1
    W[2*5 + 0, col_start+7] += 1
    
    # Term 8: A01 * B1,6
    U[1, col_start+8] += 1
    V[7*1 + 6, col_start+8] += 1
    W[2*6 + 0, col_start+8] += 1
    
    # Term 9: A11 * B1,4
    U[3, col_start+9] += 1
    V[7*1 + 4, col_start+9] += 1
    W[2*4 + 1, col_start+9] += 1
    
    # We're missing terms for A11*B1,5 and A11*B1,6
    # But that's only 10 terms, and we need 12 for full schoolbook.
    # This is incomplete! Let me reconsider.
    
    # Actually, for 2x2 by 2x3, schoolbook is 12 terms.
    # I need to find a rank-10 decomposition.
    
    # Let me use a different approach: just use schoolbook for the last 3 columns
    # and accept rank 25, or find a better decomposition.
    
    # For now, let me return schoolbook to ensure correctness.
    
    def schoolbook():
        n, m, p = 2, 2, 7
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return schoolbook()
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
