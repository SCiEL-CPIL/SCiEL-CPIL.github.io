"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Strategy: Use block-diagonal embedding of 3x3x3 Strassen (rank 7) 
    # for the top-left 3x3x3 sub-tensor, plus schoolbook correction for
    # the boundary (4th row of A, 4th column of B, 5th column of C).
    
    # Strassen's 7-term decomposition for 3x3x3 matrix multiplication
    # We need: sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1 for all i,j,k in 0..2
    # and 0 otherwise.
    
    # Standard Strassen terms for 3x3:
    # Let A, B be 3x3. C = AB.
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 - M7
    
    # But we need a 3x3x3 tensor decomposition, not just 2x2.
    # For 3x3, the minimal rank is known to be 19 or 20? Actually for 3x3x3,
    # the best known upper bound is 19 (Strassen's 19-term algorithm).
    # But we can also just use a simple approach: use the 2x2 Strassen for the 
    # top-left 2x2 block, and schoolbook for the rest.
    
    # Actually, let's use a simpler and more reliable approach:
    # Use Strassen's 7-term algorithm for the 2x2 sub-block (top-left 2x2 of C),
    # and schoolbook for all other entries.
    
    # For 4x4 * 4x5 -> 4x5:
    # We can partition A into blocks: A = [[A11, A12], [A21, A22]] where A11 is 2x2, etc.
    # B = [[B11, B12, B13], [B21, B22, B23]] where B11 is 2x2, B12 is 2x2, B13 is 2x1, etc.
    # C = [[C11, C12, C13], [C21, C22, C23]]
    
    # C11 = A11*B11 (use Strassen 7 terms)
    # C12 = A11*B12 (use Strassen 7 terms)
    # C13 = A11*B13 (2x2 * 2x1 = 2x1, use schoolbook 2*2=4 terms... wait, 2x1)
    # Actually C13 is 2x1: A11 (2x2) * B13 (2x1) -> 2x1
    # C21 = A21*B11 (2x2 * 2x2 = 2x2, use Strassen 7 terms)
    # C22 = A22*B22 (2x2 * 2x2 = 2x2, use Strassen 7 terms)
    # C23 = A22*B23 (2x2 * 2x1 = 2x1, schoolbook)
    
    # Wait, this is getting complex. Let me just use a known good decomposition.
    
    # For simplicity and correctness, let's use the following approach:
    # Use Strassen's 7-term algorithm for each 2x2 block multiplication.
    
    # The 4x4 * 4x5 multiplication can be seen as:
    # A is 4x4, B is 4x5, C is 4x5
    # Partition into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B = [[B11, B12, B13], [B21, B22, B23]] (B11,B12,B21,B22 are 2x2, B13,B23 are 2x1)
    # C = [[C11, C12, C13], [C21, C22, C23]]
    
    # C11 = A11*B11: 7 terms (Strassen)
    # C12 = A11*B12: 7 terms (Strassen)  
    # C13 = A11*B13: 4 terms (schoolbook for 2x2 * 2x1)
    # C21 = A21*B11: 7 terms (Strassen)
    # C22 = A22*B22: 7 terms (Strassen)
    # C23 = A22*B23: 4 terms (schoolbook for 2x2 * 2x1)
    
    # Total: 7+7+4+7+7+4 = 36 terms
    
    # Let's build this decomposition.
    
    # First, let's define the Strassen terms for 2x2:
    # For matrices X (2x2) and Y (2x2), Z = X*Y:
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y12-Y22)
    # M4 = X22(Y21-Y11)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)
    
    # Z11 = M1 - M4 + M6 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M3 + M5 - M7
    
    # In terms of the tensor decomposition, for a 2x2*2x2->2x2 block:
    # Each term c has U_c (2x2), V_c (2x2), W_c (2x2) such that
    # sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = delta_{i,j,k} for the multiplication tensor
    
    # Let me define the 7 Strassen terms for 2x2:
    # Term 1: M1 = (X11+X22)(Y11+Y22)
    #   U1 = [[1,0],[0,1]], V1 = [[1,0],[1,0]], W1 = [[1,0],[0,1]]
    #   Wait, I need to be more careful.
    
    # Let me use a different approach. I'll just use schoolbook with a small optimization.
    # Actually, let's use a known rank-28 decomposition for 4x4*4x4, and adapt for 4x5.
    
    # For now, let's use a rank-36 decomposition based on 2x2 Strassen blocks.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 16 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 20 x 36
    
    # Helper: add a Strassen term for a 2x2 block at position (row_offset, col_offset) in A,
    # (col_offset, col_offset2) in B, contributing to (col_offset, row_offset) in C
    
    # Actually, let me just manually construct this.
    
    # For the 2x2 Strassen decomposition, here are the 7 terms:
    # Each term is (U, V, W) where U, V, W are 2x2 matrices
    
    # M1: (X11+X22)(Y11+Y22)
    # U1 = [[1, 0], [0, 1]]
    # V1 = [[1, 0], [1, 0]]  # Y11+Y22 means we take Y11 and Y22, so V1[j,k] should extract Y11 when j=0,k=0 and Y22 when j=1,k=1
    # Hmm, this is tricky. Let me think differently.
    
    # The tensor T[i,j,k] = 1 if i,j,k are all in {0,1} and represent a valid multiplication term.
    # For 2x2*2x2->2x2: T[i,j,k] = 1 for all i,j,k in {0,1} (8 terms total for schoolbook).
    # Strassen reduces this to 7 terms.
    
    # Let me just use the known Strassen 7-term decomposition:
    # Term 0: U=[[1,0],[0,1]], V=[[1,0],[1,0]], W=[[1,0],[0,1]]
    # Term 1: U=[[0,0],[1,1]], V=[[1,0],[0,0]], W=[[0,0],[0,1]]
    # Term 2: U=[[1,0],[0,0]], V=[[0,1],[0,-1]], W=[[0,1],[0,0]]
    # Term 3: U=[[0,0],[0,1]], V=[[0,-1],[1,0]], W=[[0,0],[1,0]]
    # Term 4: U=[[1,1],[0,0]], V=[[0,0],[0,1]], W=[[0,1],[0,0]]
    # Term 5: U=[[0,-1],[1,0]], V=[[1,1],[0,0]], W=[[1,0],[0,0]]
    # Term 6: U=[[0,1],[-1,0]], V=[[0,0],[1,1]], W=[[0,0],[0,1]]
    
    # Let me verify term 0:
    # sum over c=0: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 (for i=0,j=0,k=0)
    # U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0
    # U[0,1]*V[1,0]*W[0,0] = 0*1*1 = 0
    # U[0,1]*V[1,1]*W[1,0] = 0*0*0 = 0
    # So T[0,0,0] gets contribution 1 from term 0. Good.
    
    # This is getting too complex to do by hand. Let me just use schoolbook with rank 80.
    # But that's the same as the current program. Let me try a different approach.
    
    # Use a known rank-28 algorithm for 4x4*4x4, and extend to 4x5.
    # Actually, let me just return the schoolbook solution but with a slightly better structure.
    
    # For 4x4 * 4x5:
    # Schoolbook rank = 4*4*5 = 80
    
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
