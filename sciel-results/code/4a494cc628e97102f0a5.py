import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose 5x5 into:
    # A = [[A44, A41],
    #      [A14, A11]]
    # B = [[B44, B41],
    #      [B14, B11]]
    # C = A B
    # C44 = A44 B44 + A41 B14
    # C41 = A44 B41 + A41 B11
    # C14 = A14 B44 + A11 B14
    # C11 = A14 B41 + A11 B11
    
    # We use Strassen for A44 B44 (4x4). Rank 49.
    # We use schoolbook for the cross terms.
    # A41 B14: (4x1) * (1x4) -> 4x4. Rank 16.
    # A44 B41: (4x4) * (4x1) -> 4x1. Rank 16.
    # A14 B44: (1x4) * (4x4) -> 1x4. Rank 16.
    # A41 B11: (4x1) * (1x1) -> 4x1. Rank 4.
    # A14 B41: (1x4) * (4x1) -> 1x1. Rank 4.
    # A11 B11: (1x1) * (1x1) -> 1x1. Rank 1.
    
    # Wait, A41 is 4x1. B14 is 1x4.
    # A41 B14 is 4x4.
    # The rank of (4x1) * (1x4) is 4. Not 16.
    # Let's re-evaluate.
    # A41 is a column vector. B14 is a row vector.
    # A41 B14 = outer product. Rank 1.
    # So A41 B14 requires 4 multiplications? No, it's 4 scalar mults, but the tensor rank is 4?
    # The tensor for (4x1) * (1x4) has rank 4.
    # Yes, because you need 4 independent multiplications.
    
    # A44 B41: (4x4) * (4x1). Result 4x1.
    # Rank is 16 (each of 4 rows of A44 dot B41).
    # A14 B44: (1x4) * (4x4). Result 1x4.
    # Rank is 16.
    
    # A41 B11: (4x1) * (1x1). Result 4x1.
    # Rank is 4.
    # A14 B41: (1x4) * (4x1). Result 1x1.
    # Rank is 4.
    # A11 B11: (1x1) * (1x1). Result 1x1.
    # Rank is 1.
    
    # Total Rank = 49 (for 4x4) + 4 (A41 B14) + 16 (A44 B41) + 16 (A14 B44) + 4 (A41 B11) + 4 (A14 B41) + 1 (A11 B11)
    # Total = 49 + 4 + 16 + 16 + 4 + 4 + 1 = 94.
    
    # This is better than 100.
    
    # Let's verify the indices.
    # A is 5x5.
    # A44: rows 0-3, cols 0-3.
    # A41: rows 0-3, col 4.
    # A14: row 4, cols 0-3.
    # A11: row 4, col 4.
    
    # B is 5x5.
    # B44: rows 0-3, cols 0-3.
    # B41: rows 0-3, col 4.
    # B14: row 4, cols 0-3.
    # B11: row 4, col 4.
    
    # C is 5x5.
    # C44: rows 0-3, cols 0-3.
    # C41: rows 0-3, col 4.
    # C14: row 4, cols 0-3.
    # C11: row 4, col 4.
    
    # C44 = A44 B44 + A41 B14
    # C41 = A44 B41 + A41 B11
    # C14 = A14 B44 + A11 B14
    # C11 = A14 B41 + A11 B11
    
    # We need to construct U, V, W.
    # U is 25 x Rank.
    # V is 25 x Rank.
    # W is 25 x Rank.
    
    rank = 94
    U = [[0] * rank for _ in range(25)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(25)]
    
    col = 0
    
    # 1. A44 B44 (Strassen 4x4)
    # We need a Strassen decomposition for 4x4.
    # We can recursively apply Strassen to 2x2 blocks.
    # Let's write a helper to get the Strassen terms for a 4x4 block.
    
    def strassen_4x4(U, V, W, offset_u, offset_v, offset_w, start_col):
        # A44 is at offset_u, B44 is at offset_v, C44 is at offset_w.
        # A44 is 4x4. Split into 2x2 blocks.
        # A11, A12, A21, A22
        # B11, B12, B21, B22
        
        # Strassen 2x2 logic:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # This gives 7 multiplications for 2x2.
        # For 4x4, we apply this to the 2x2 blocks of 2x2 matrices.
        # This results in 7 * 7 = 49 multiplications.
        
        # We need to map the 4x4 indices to the 25x25 U, V, W.
        
        # This is complex to implement inline.
        # Instead, I will use the schoolbook for 4x4 to be safe and simple, 
        # which is rank 16.
        # Then total rank = 16 + 4 + 16 + 16 + 4 + 4 + 1 = 61.
        # Wait, schoolbook 4x4 is 16.
        # 16 + 4 + 16 + 16 + 4 + 4 + 1 = 61.
        # This is much better than 125.
        
        # Let's use schoolbook for the 4x4 part.
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    # A44[i,k] * B44[k,j]
                    # U index for A44[i,k]: offset_u + i*5 + k
                    # V index for B44[k,j]: offset_v + k*5 + j
                    # W index for C44[i,j]: offset_w + i*5 + j
                    
                    U[offset_u + i*5 + k][start_col] = 1
                    V[offset_v + k*5 + j][start_col] = 1
                    W[offset_w + i*5 + j][start_col] = 1
                    start_col += 1
        return start_col
    
    # Offset for A44: 0
    # Offset for B44: 0
    # Offset for C44: 0
    col = strassen_4x4(U, V, W, 0, 0, 0, col) # Using schoolbook for 4x4 (rank 16)
    
    # 2. A41 B14 (Rank 4)
    # A41 is col 4 of A, rows 0-3.
    # B14 is row 4 of B, cols 0-3.
    # C44 gets contribution.
    # A41[i, 4] * B14[4, j] -> C44[i, j]
    for i in range(4):
        for j in range(4):
            U[i*5 + 4][col] = 1
            V[4*5 + j][col] = 1
            W[i*5 + j][col] = 1
            col += 1
            
    # 3. A44 B41 (Rank 16)
    # A44 is 4x4. B41 is col 4 of B, rows 0-3.
    # C41 gets contribution.
    # A44[i, k] * B41[k, 4] -> C41[i, 4]
    for i in range(4):
        for k in range(4):
            U[i*5 + k][col] = 1
            V[k*5 + 4][col] = 1
            W[i*5 + 4][col] = 1
            col += 1
            
    # 4. A14 B44 (Rank 16)
    # A14 is row 4 of A, cols 0-3.
    # B44 is 4x4.
    # C14 gets contribution.
    # A14[4, k] * B44[k, j] -> C14[4, j]
    for k in range(4):
        for j in range(4):
            U[4*5 + k][col] = 1
            V[k*5 + j][col] = 1
            W[4*5 + j][col] = 1
            col += 1
            
    # 5. A41 B11 (Rank 4)
    # A41 is col 4 of A, rows 0-3.
    # B11 is B[4,4].
    # C41 gets contribution.
    # A41[i, 4] * B11[4, 4] -> C41[i, 4]
    for i in range(4):
        U[i*5 + 4][col] = 1
        V[4*5 + 4][col] = 1
        W[i*5 + 4][col] = 1
        col += 1
        
    # 6. A14 B41 (Rank 4)
    # A14 is row 4 of A, cols 0-3.
    # B41 is col 4 of B, rows 0-3.
    # C11 gets contribution.
    # A14[4, k] * B41[k, 4] -> C11[4, 4]
    for k in range(4):
        U[4*5 + k][col] = 1
        V[k*5 + 4][col] = 1
        W[4*5 + 4][col] = 1
        col += 1
        
    # 7. A11 B11 (Rank 1)
    # A11 is A[4,4].
    # B11 is B[4,4].
    # C11 gets contribution.
    # A11[4, 4] * B11[4, 4] -> C11[4, 4]
    U[4*5 + 4][col] = 1
    V[4*5 + 4][col] = 1
    W[4*5 + 4][col] = 1
    col += 1
    
    # Total rank is 16 + 4 + 16 + 16 + 4 + 4 + 1 = 61.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
