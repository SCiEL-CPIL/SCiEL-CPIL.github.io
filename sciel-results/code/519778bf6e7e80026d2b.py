"""Matrix multiplication 2x3x8 tensor decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose 2x3x8 as two independent 2x3x4 blocks.
    # Each 2x3x4 block uses rank 11 (best known for 2x3x2 scaled appropriately).
    # Total rank = 2 * 11 = 22
    rank = 22

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # For each half of the 8 columns (k=0..3 and k=4..7),
    # we decompose the 2x3x4 multiplication.
    # 2x3x4 can be viewed as: A(2x3) * B(3x4)
    # Split B into B1(3x2) and B2(3x2), each giving 2x3x2.
    # 2x3x2 has rank 11 (best known).
    # So 2x3x4 = 2x3x2 + 2x3x2 with rank 11+11=22 per half? No, that's 44.
    
    # Alternative: treat 2x3x4 directly.
    # 2x3x4 tensor: A has 6 entries, B has 12 entries, C has 8 entries.
    # Schoolbook rank = 6*12/8 = 9? No, schoolbook is 2*3*4=24.
    # Best known for 2x3x4: we can use 2x2x4 + 2x1x4.
    # 2x2x4: rank 11 (Strassen-like for 2x2x2 blocks: 2*7+4=18? No)
    
    # Let me use a known approach: 2x3x4 = 2x2x4 + 2x1x4
    # 2x2x4: split into two 2x2x2 blocks, each rank 7 (Strassen) + 4 (for 2x1x2)
    # Actually 2x2x2 has rank 7 (Strassen), so 2x2x4 = 2*7 = 14? No, that's not right either.
    
    # Let me use the original approach but more carefully.
    # Original: 4 blocks of 2x2x2 (Strassen, rank 7 each) + 4 blocks of 2x1x2 (rank 4 each)
    # = 4*7 + 4*4 = 28 + 16 = 44
    
    # New approach: 2 blocks of 2x3x4.
    # For 2x3x4: A is 2x3, B is 3x4, C is 2x4.
    # Split A into A1(2x2) and A2(2x1), B into B1(3x2) and B2(3x2).
    # C1 = A1*B1 (2x2x2, rank 7) + A1*B2 (2x2x2, rank 7) + A2*B1 (2x1x2, rank 4) + A2*B2 (2x1x2, rank 4)
    # Total per 2x3x4: 7+7+4+4 = 22. Two blocks: 44. Same as before.
    
    # Let me try to find a better decomposition for 2x3x2.
    # 2x3x2: A(2x3), B(3x2), C(2x2).
    # Schoolbook: 2*3*2 = 12.
    # Best known: 11.
    # If 2x3x2 has rank 11, then 2x3x4 = 2x3x2 + 2x3x2 = 22, and 2x3x8 = 4*22 = 88. Worse.
    
    # I'll stick with the original rank 44 approach but write it cleanly.
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0][col] = 1; U[4][col] = 1
        V[k0][col] = 1; V[p + k1][col] = 1
        W[k0][col] = 1; W[k1 + n][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B0k0
        U[3][col] = 1; U[4][col] = 1
        V[k0][col] = 1
        W[k0 + n][col] = 1; W[k1 + n][col] = -1
        col += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0][col] = 1
        V[k1][col] = 1; V[p + k1][col] = -1
        W[k1][col] = 1; W[k1 + n][col] = 1
        col += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[4][col] = 1
        V[p + k0][col] = 1; V[k0][col] = -1
        W[k0][col] = 1; W[k0 + n][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0][col] = 1; U[1][col] = 1
        V[p + k1][col] = 1
        W[k0][col] = -1; W[k1][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[3][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[k1][col] = 1
        W[k1 + n][col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1][col] = 1; U[4][col] = -1
        V[p + k0][col] = 1; V[p + k1][col] = 1
        W[k0][col] = 1
        col += 1
        
        # P1-P4: 2x1x2 part
        U[2][col] = 1; V[2 * p + k0][col] = 1; W[k0][col] = 1
        col += 1
        U[2][col] = 1; V[2 * p + k1][col] = 1; W[k1][col] = 1
        col += 1
        U[5][col] = 1; V[2 * p + k0][col] = 1; W[k0 + n][col] = 1
        col += 1
        U[5][col] = 1; V[2 * p + k1][col] = 1; W[k1 + n][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
