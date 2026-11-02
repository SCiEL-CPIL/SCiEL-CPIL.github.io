"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will construct a rank 72 decomposition.
    # Structure:
    # Split A (4x4) into 4 blocks of 2x2: A00, A01, A10, A11
    # Split B (4x5) into 2 blocks of 2x5: B0 (top), B1 (bot)
    # C = [A00*B0 + A01*B1; A10*B0 + A11*B1]
    # Each term is 2x2 * 2x5.
    # 2x2 * 2x5 is split into 2x2*2x2 + 2x2*2x2 + 2x2*2x1.
    # Ranks: 7 + 7 + 4 = 18.
    # Total rank: 4 * 18 = 72.
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)] # 16 x 72
    V = [[0] * rank for _ in range(m * p)] # 20 x 72
    W = [[0] * rank for _ in range(p * n)] # 20 x 72
    
    # Helper to add a Strassen 2x2x2 block
    # Indices for A block: rows r0, r0+1; cols c0, c0+1
    # Indices for B block: rows r0, r0+1; cols c0, c0+1
    # Indices for C block: rows r0, r0+1; cols c0, c0+1
    # Note: C indices are (col, row) in the output matrix?
    # Task says W[k*n+i, r]. So index is k*n + i.
    # C is p x n.
    # B is m x p.
    # A is n x m.
    
    # Let's map indices carefully.
    # A[i, j] -> U[i*m + j, col]
    # B[j, k] -> V[j*p + k, col]
    # C[k, i] -> W[k*n + i, col]
    
    col = 0
    
    # Strassen 2x2x2 coefficients
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    def add_strassen(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0):
        # A_r0, A_c0: top-left of A block
        # B_r0, B_c0: top-left of B block
        # C_r0, C_c0: top-left of C block (C is p x n, so C_r0 is k, C_c0 is i)
        # Note: In C, rows are k, cols are i.
        # In A, rows are i, cols are j.
        # In B, rows are j, cols are k.
        
        nonlocal col
        # M1
        U[A_r0*m + A_c0][col] += 1
        U[(A_r0+1)*m + (A_c0+1)][col] += 1
        V[B_r0*p + B_c0][col] += 1
        V[(B_r0+1)*p + (B_c0+1)][col] += 1
        W[C_r0*n + C_c0][col] += 1
        W[(C_r0+1)*n + (C_c0+1)][col] += 1
        col += 1
        
        # M2
        U[(A_r0+1)*m + A_c0][col] += 1
        U[(A_r0+1)*m + (A_c0+1)][col] += 1
        V[B_r0*p + B_c0][col] += 1
        W[(C_r0+1)*n + C_c0][col] += 1
        W[(C_r0+1)*n + (C_c0+1)][col] -= 1
        col += 1
        
        # M3
        U[A_r0*m + A_c0][col] += 1
        V[B_r0*p + (B_c0+1)][col] += 1
        V[(B_r0+1)*p + (B_c0+1)][col] -= 1
        W[C_r0*n + (C_c0+1)][col] += 1
        W[(C_r0+1)*n + (C_c0+1)][col] += 1
        col += 1
        
        # M4
        U[(A_r0+1)*m + (A_c0+1)][col] += 1
        V[(B_r0+1)*p + B_c0][col] += 1
        V[B_r0*p + B_c0][col] -= 1
        W[C_r0*n + C_c0][col] += 1
        W[(C_r0+1)*n + C_c0][col] += 1
        col += 1
        
        # M5
        U[A_r0*m + A_c0][col] += 1
        U[A_r0*m + (A_c0+1)][col] += 1
        V[(B_r0+1)*p + (B_c0+1)][col] += 1
        W[C_r0*n + C_c0][col] -= 1
        W[C_r0*n + (C_c0+1)][col] += 1
        col += 1
        
        # M6
        U[(A_r0+1)*m + A_c0][col] += 1
        U[A_r0*m + A_c0][col] -= 1
        V[B_r0*p + B_c0][col] += 1
        V[B_r0*p + (B_c0+1)][col] += 1
        W[(C_r0+1)*n + (C_c0+1)][col] += 1
        col += 1
        
        # M7
        U[A_r0*m + (A_c0+1)][col] += 1
        U[(A_r0+1)*m + (A_c0+1)][col] -= 1
        V[(B_r0+1)*p + B_c0][col] += 1
        V[(B_r0+1)*p + (B_c0+1)][col] += 1
        W[C_r0*n + C_c0][col] += 1
        col += 1

    def add_trivial_2x2x1(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0):
        # A 2x2, B 2x1, C 2x1
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        nonlocal col
        # Term 1: A00 B0 -> C0
        U[A_r0*m + A_c0][col] += 1
        V[B_r0*p + B_c0][col] += 1
        W[C_r0*n + C_c0][col] += 1
        col += 1
        # Term 2: A01 B1 -> C0
        U[A_r0*m + (A_c0+1)][col] += 1
        V[(B_r0+1)*p + B_c0][col] += 1
        W[C_r0*n + C_c0][col] += 1
        col += 1
        # Term 3: A10 B0 -> C1
        U[(A_r0+1)*m + A_c0][col] += 1
        V[B_r0*p + B_c0][col] += 1
        W[(C_r0+1)*n + C_c0][col] += 1
        col += 1
        # Term 4: A11 B1 -> C1
        U[(A_r0+1)*m + (A_c0+1)][col] += 1
        V[(B_r0+1)*p + B_c0][col] += 1
        W[(C_r0+1)*n + C_c0][col] += 1
        col += 1

    # Block 00: A[0:2, 0:2] * B[0:2, 0:5] -> C[0:2, 0:2]
    # B[0:2, 0:5] split into B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:5]
    add_strassen(0, 0, 0, 0, 0, 0) # A00*B00 -> C00
    add_strassen(0, 0, 0, 2, 0, 2) # A00*B02 -> C02 (Wait, C is 2x2 block, so C cols 0,1)
    # Correction: C block is 2x2. C_r0, C_c0 are indices in C.
    # C[0:2, 0:2] corresponds to k=0,1 and i=0,1.
    # B[0:2, 0:2] is j=0,1, k=0,1.
    # B[0:2, 2:4] is j=0,1, k=2,3.
    # B[0:2, 4:5] is j=0,1, k=4.
    
    # We need to be careful with C indices.
    # C block 00 is k=0,1 and i=0,1.
    # So C_r0=0, C_c0=0.
    # The second part of B (k=2,3) contributes to C[k=2,3, i=0,1].
    # This is C block 02? No, C is 5x4.
    # C rows are k=0..4. C cols are i=0..3.
    # C block 00 is k=0,1; i=0,1.
    # C block 01 is k=0,1; i=2,3.
    # C block 10 is k=2,3; i=0,1.
    # C block 11 is k=2,3; i=2,3.
    # C block 12 is k=4; i=0,1.
    # C block 13 is k=4; i=2,3.
    
    # So for A00 * B0 (k=0,1), it contributes to C[0:2, 0:2].
    # For A00 * B2 (k=2,3), it contributes to C[2:4, 0:2].
    # For A00 * B4 (k=4), it contributes to C[4:5, 0:2].
    
    # Let's restart the block mapping.
    # A blocks: A00 (i=0,1, j=0,1), A01 (i=0,1, j=2,3), A10 (i=2,3, j=0,1), A11 (i=2,3, j=2,3).
    # B blocks: B0 (j=0,1, k=0,1), B1 (j=0,1, k=2,3), B2 (j=0,1, k=4), B3 (j=2,3, k=0,1), B4 (j=2,3, k=2,3), B5 (j=2,3, k=4).
    
    # C = A00 B0 + A00 B1 + A00 B2 + A01 B3 + A01 B4 + A01 B5 + A10 B0 + ...
    
    # We will process each A block and its corresponding B blocks.
    
    # A00 (i=0,1, j=0,1)
    # Multiplies B0 (j=0,1, k=0,1) -> C[0:2, 0:2] (k=0,1, i=0,1)
    add_strassen(0, 0, 0, 0, 0, 0)
    # Multiplies B1 (j=0,1, k=2,3) -> C[2:4, 0:2] (k=2,3, i=0,1)
    add_strassen(0, 0, 0, 2, 2, 0)
    # Multiplies B2 (j=0,1, k=4) -> C[4:5, 0:2] (k=4, i=0,1)
    add_trivial_2x2x1(0, 0, 0, 4, 4, 0)
    
    # A01 (i=0,1, j=2,3)
    # Multiplies B3 (j=2,3, k=0,1) -> C[0:2, 0:2]
    add_strassen(0, 2, 2, 0, 0, 0)
    # Multiplies B4 (j=2,3, k=2,3) -> C[2:4, 0:2]
    add_strassen(0, 2, 2, 2, 2, 0)
    # Multiplies B5 (j=2,3, k=4) -> C[4:5, 0:2]
    add_trivial_2x2x1(0, 2, 2, 4, 4, 0)
    
    # A10 (i=2,3, j=0,1)
    # Multiplies B0 (j=0,1, k=0,1) -> C[0:2, 2:4] (k=0,1, i=2,3)
    add_strassen(2, 0, 0, 0, 0, 2)
    # Multiplies B1 (j=0,1, k=2,3) -> C[2:4, 2:4]
    add_strassen(2, 0, 0, 2, 2, 2)
    # Multiplies B2 (j=0,1, k=4) -> C[4:5, 2:4]
    add_trivial_2x2x1(2, 0, 0, 4, 4, 2)
    
    # A11 (i=2,3, j=2,3)
    # Multiplies B3 (j=2,3, k=0,1) -> C[0:2, 2:4]
    add_strassen(2, 2, 2, 0, 0, 2)
    # Multiplies B4 (j=2,3, k=2,3) -> C[2:4, 2:4]
    add_strassen(2, 2, 2, 2, 2, 2)
    # Multiplies B5 (j=2,3, k=4) -> C[4:5, 2:4]
    add_trivial_2x2x1(2, 2, 2, 4, 4, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
