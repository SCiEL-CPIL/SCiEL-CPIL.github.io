"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = 2, 3, 6
    deadline = time.time() + 140  # safety margin
    
    # Rank 30 decomposition:
    # Partition B (3x6) into three 3x2 blocks: B0, B1, B2
    # For each 2x3 * 3x2 block, use rank-9 decomposition:
    #   - Strassen 2x2x2 for first two columns (7 mults)
    #   - Direct multiplication for third column (2 mults)
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]  # 6 x 30
    V = [[0] * rank for _ in range(m * p)]  # 18 x 30
    W = [[0] * rank for _ in range(p * n)]  # 12 x 30
    
    def strassen_2x2x2(a_base, b_base, c_base, col_start):
        """
        Add 7 Strassen terms for 2x2 * 2x2.
        A indices: a_base, a_base+1 (row 0), a_base+2, a_base+3 (row 1)
        B indices: b_base, b_base+1 (col 0), b_base+2, b_base+3 (col 1)
        C indices: c_base, c_base+1 (row 0), c_base+2, c_base+3 (row 1)
        """
        # m1 = (A00+A11)(B00+B11)
        U[a_base][col_start] = 1
        U[a_base+2][col_start] = 1
        V[b_base][col_start] = 1
        V[b_base+2][col_start] = 1
        W[c_base][col_start] = 1
        W[c_base+2][col_start] = 1
        W[c_base+1][col_start] = -1
        W[c_base+3][col_start] = -1
        
        # m2 = (A10+A11)B00
        U[a_base+2][col_start+1] = 1
        U[a_base+3][col_start+1] = 1
        V[b_base][col_start+1] = 1
        W[c_base+1][col_start+1] = 1
        W[c_base+3][col_start+1] = 1
        
        # m3 = A00(B10-B11)
        U[a_base][col_start+2] = 1
        V[b_base+1][col_start+2] = 1
        V[b_base+3][col_start+2] = -1
        W[c_base][col_start+2] = 1
        W[c_base+2][col_start+2] = 1
        
        # m4 = A11(B00-B01)
        U[a_base+3][col_start+3] = 1
        V[b_base][col_start+3] = 1
        V[b_base+1][col_start+3] = -1
        W[c_base+1][col_start+3] = 1
        W[c_base+3][col_start+3] = 1
        
        # m5 = (A00+A01)B11
        U[a_base][col_start+4] = 1
        U[a_base+1][col_start+4] = 1
        V[b_base+2][col_start+4] = 1
        W[c_base][col_start+4] = 1
        W[c_base+2][col_start+4] = 1
        
        # m6 = (A10-A00)(B00+B01)
        U[a_base+2][col_start+5] = 1
        U[a_base][col_start+5] = -1
        V[b_base][col_start+5] = 1
        V[b_base+1][col_start+5] = 1
        W[c_base][col_start+5] = -1
        W[c_base+2][col_start+5] = 1
        
        # m7 = (A01-A11)(B10+B11)
        U[a_base+1][col_start+6] = 1
        U[a_base+3][col_start+6] = -1
        V[b_base+1][col_start+6] = 1
        V[b_base+3][col_start+6] = 1
        W[c_base+1][col_start+6] = -1
        W[c_base+3][col_start+6] = 1
        
        return col_start + 7
    
    col = 0
    
    # Process three 3x2 blocks of B
    for block in range(3):
        # B block: rows j=0,1,2; cols k=2*block, 2*block+1
        # A: 2x3, indices:
        #   Row 0: j=0,1,2 -> indices 0,1,2
        #   Row 1: j=0,1,2 -> indices 3,4,5
        # B block: 
        #   Col 0: j=0,1,2 -> indices 2*block, 2*block+1, 2*block+2
        #   Col 1: j=0,1,2 -> indices 2*block+3, 2*block+4, 2*block+5
        # C: 2x2, indices:
        #   Row 0: k=2*block, 2*block+1 -> indices 2*block, 2*block+1
        #   Row 1: k=2*block, 2*block+1 -> indices 2*block+2, 2*block+3
        
        # Split A into top (1x3) and bottom (1x3)
        # Top: A[0,j] for j=0,1,2 -> U indices 0,1,2
        # Bottom: A[1,j] for j=0,1,2 -> U indices 3,4,5
        
        # B block:
        #   B[0,k] for k=0,1 -> V indices 2*block, 2*block+3
        #   B[1,k] for k=0,1 -> V indices 2*block+1, 2*block+4
        #   B[2,k] for k=0,1 -> V indices 2*block+2, 2*block+5
        
        # C block:
        #   C[0,k] for k=0,1 -> W indices 2*block, 2*block+1
        #   C[1,k] for k=0,1 -> W indices 2*block+2, 2*block+3
        
        # Use Strassen on the 2x2x2 core (j=0,1 for A and B, k=0,1 for C)
        # A core: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,3,4
        # B core: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices 2*block, 2*block+3, 2*block+1, 2*block+4
        # C core: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices 2*block, 2*block+1, 2*block+2, 2*block+3
        
        # For Strassen, we need:
        # A00 = A[0,0] = U[0]
        # A01 = A[0,1] = U[1]
        # A10 = A[1,0] = U[3]
        # A11 = A[1,1] = U[4]
        # B00 = B[0,0] = V[2*block]
        # B01 = B[0,1] = V[2*block+3]
        # B10 = B[1,0] = V[2*block+1]
        # B11 = B[1,1] = V[2*block+4]
        # C00 = C[0,0] = W[2*block]
        # C01 = C[0,1] = W[2*block+1]
        # C10 = C[1,0] = W[2*block+2]
        # C11 = C[1,1] = W[2*block+3]
        
        # Strassen mapping:
        # m1 = (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11
        # m2 = (A10+A11)B00 -> contributes to C01, C11
        # m3 = A00(B10-B11) -> contributes to C00, C10
        # m4 = A11(B00-B01) -> contributes to C01, C11
        # m5 = (A00+A01)B11 -> contributes to C00, C10
        # m6 = (A10-A00)(B00+B01) -> contributes to C00, C10
        # m7 = (A01-A11)(B10+B11) -> contributes to C01, C11
        
        # But we also need to account for A[0,2] and A[1,2] multiplying B[2,0] and B[2,1]
        # These give additional contributions to C[0,0], C[0,1], C[1,0], C[1,1]
        
        # For now, let's just do the 7 Strassen mults for the 2x2x2 part
        # and 2 more for the j=2 part
        
        # Strassen for j=0,1 part
        a_base = 0  # A[0,0]
        # But we need to map correctly
        # Let me redo the Strassen with explicit indices
        
        # m1 = (A00+A11)(B00+B11)
        U[0][col] = 1; U[4][col] = 1
        V[2*block][col] = 1; V[2*block+4][col] = 1
        W[2*block][col] = 1; W[2*block+1][col] = 1
        W[2*block+2][col] = 1; W[2*block+3][col] = 1
        col += 1
        
        # m2 = (A10+A11)B00
        U[3][col] = 1; U[4][col] = 1
        V[2*block][col] = 1
        W[2*block+1][col] = 1; W[2*block+3][col] = 1
        col += 1
        
        # m3 = A00(B10-B11)
        U[0][col] = 1
        V[2*block+1][col] = 1; V[2*block+5][col] = -1
        W[2*block][col] = 1; W[2*block+2][col] = 1
        col += 1
        
        # m4 = A11(B00-B01)
        U[4][col] = 1
        V[2*block][col] = 1; V[2*block+3][col] = -1
        W[2*block+1][col] = 1; W[2*block+3][col] = 1
        col += 1
        
        # m5 = (A00+A01)B11
        U[0][col] = 1; U[1][col] = 1
        V[2*block+4][col] = 1
        W[2*block][col] = 1; W[2*block+2][col] = 1
        col += 1
        
        # m6 = (A10-A00)(B00+B01)
        U[3][col] = 1; U[0][col] = -1
        V[2*block][col] = 1; V[2*block+3][col] = 1
        W[2*block][col] = -1; W[2*block+2][col] = 1
        col += 1
        
        # m7 = (A01-A11)(B10+B11)
        U[1][col] = 1; U[4][col] = -1
        V[2*block+1][col] = 1; V[2*block+5][col] = 1
        W[2*block+1][col] = -1; W[2*block+3][col] = 1
        col += 1
        
        # Now handle j=2 part: A[0,2]*B[2,k] and A[1,2]*B[2,k]
        # m8 = A[0,2] * B[2,0]
        U[2][col] = 1
        V[2*block+2][col] = 1
        W[2*block][col] = 1
        col += 1
        
        # m9 = A[0,2] * B[2,1]
        U[2][col] = 1
        V[2*block+5][col] = 1
        W[2*block+1][col] = 1
        col += 1
        
        # m10 = A[1,2] * B[2,0]
        U[5][col] = 1
        V[2*block+2][col] = 1
        W[2*block+2][col] = 1
        col += 1
        
        # m11 = A[1,2] * B[2,1]
        U[5][col] = 1
        V[2*block+5][col] = 1
        W[2*block+3][col] = 1
        col += 1
    
    # Verify: we should have used 11 mults per block * 3 blocks = 33, but rank is 30
    # This means my decomposition is wrong. Let me reconsider.
    
    # Actually, the Strassen 2x2x2 gives 7 mults, and the j=2 part needs 4 more (not 2)
    # because we have A[0,2] and A[1,2] each multiplying B[2,0] and B[2,1]
    # So 7 + 4 = 11 per block, 33 total. That's too many.
    
    # Let me reconsider the approach. Maybe I should use a different decomposition.
    
    # For now, return schoolbook rank 36 to be safe
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
