"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a hybrid approach:
    # 1. Strassen for the top-left 4x4 submatrix (Rank 49)
    # 2. Schoolbook for the remaining terms involving the 5th row/col
    
    # Total Rank
    # 4x4 Strassen: 49
    # C[0:4, 4] = A[0:4, 0:4] @ B[0:4, 4] (4 mults) + A[0:4, 4] * B[4, 4] (4 mults)
    # C[4, 0:4] = A[4, 0:4] @ B[0:4, 0:4] (4 mults) + A[4, 4] * B[4, 0:4] (4 mults)
    # C[4, 4]   = A[4, 0:4] @ B[0:4, 4] (4 mults) + A[4, 4] * B[4, 4] (1 mult)
    # Total Rank = 49 + 4 + 4 + 4 + 4 + 4 + 1 = 70
    
    rank = 70
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: Strassen for 4x4 (indices 0-3) ---
    # We need to decompose 4x4x4 multiplication into 49 rank-1 terms.
    # Strassen for 2x2 uses 7 multiplications.
    # For 4x4, we split into four 2x2 blocks.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    
    # Strassen factors for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We will generate the 49 terms by iterating over the 7x7 combinations of the 2x2 Strassen steps?
    # Actually, it's easier to just hardcode the 49 terms or generate them recursively.
    # Let's use a recursive helper to get the 49 terms for 4x4.
    
    def get_strassen_4x4():
        terms = []
        # Indices for 4x4:
        # A: (0..3, 0..3) -> 16 elements
        # B: (0..3, 0..3) -> 16 elements
        # C: (0..3, 0..3) -> 16 elements
        
        # We can use the fact that 4x4 multiplication is equivalent to 7 multiplications of 2x2 matrices.
        # Each 2x2 multiplication has rank 7.
        # So we have 7 "macro" multiplications, each decomposed into 7 "micro" multiplications.
        # Total 49.
        
        # Let's define the 2x2 Strassen decomposition as a list of (u, v, w) tensors.
        # u, v, w are vectors of length 4 (for 2x2 matrices).
        # But we need to map them to 4x4 indices.
        
        # Instead of complex mapping, we can just use the schoolbook for 4x4 if it's simpler?
        # Schoolbook for 4x4 is rank 64.
        # Strassen is 49.
        # 64 + 16 (remaining) = 80.
        # 49 + 16 = 65.
        # 65 < 80.
        # So Strassen is better.
        
        # Let's implement the 4x4 Strassen explicitly.
        # This is tedious to hardcode 49 terms.
        # Alternative: Use the fact that R(4) <= 49 is well known, but coding it is error-prone.
        # What if I just use Schoolbook for 4x4 (Rank 64) and the rest (16 terms)?
        # Total Rank = 64 + 16 = 80.
        # 80 < 125.
        # This is much safer to code correctly.
        
        return 80 # Just a placeholder to decide

    # Let's stick to Schoolbook for the 4x4 part to ensure correctness.
    # Rank for 4x4x4 schoolbook is 64.
    # Rank for remaining terms:
    # C[0:4, 4]: 4 (A[0:4, 0:4]@B[0:4, 4]) + 4 (A[0:4, 4]*B[4, 4]) = 8
    # C[4, 0:4]: 4 (A[4, 0:4]@B[0:4, 0:4]) + 4 (A[4, 4]*B[4, 0:4]) = 8
    # C[4, 4]: 4 (A[4, 0:4]@B[0:4, 4]) + 1 (A[4, 4]*B[4, 4]) = 5
    # Total Rank = 64 + 8 + 8 + 5 = 85.
    
    # 85 is better than 125.
    # Let's implement Rank 85.
    
    rank = 85
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: Schoolbook for 4x4 (indices 0-3)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # A[i, j] * B[j, k] -> C[i, k]
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 5 + i][col] = 1
                col += 1
                
    # Part 2: C[0:4, 4] = A[0:4, 0:4] @ B[0:4, 4]
    for i in range(4):
        for j in range(4):
            # A[i, j] * B[j, 4] -> C[i, 4]
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 5 + i][col] = 1
            col += 1
            
    # Part 3: C[0:4, 4] += A[0:4, 4] * B[4, 4]
    for i in range(4):
        # A[i, 4] * B[4, 4] -> C[i, 4]
        U[i * 5 + 4][col] = 1
        V[4 * 5 + 4][col] = 1
        W[4 * 5 + i][col] = 1
        col += 1
        
    # Part 4: C[4, 0:4] = A[4, 0:4] @ B[0:4, 0:4]
    for j in range(4):
        for k in range(4):
            # A[4, j] * B[j, k] -> C[4, k]
            U[4 * 5 + j][col] = 1
            V[j * 5 + k][col] = 1
            W[k * 5 + 4][col] = 1
            col += 1
            
    # Part 5: C[4, 0:4] += A[4, 4] * B[4, 0:4]
    for k in range(4):
        # A[4, 4] * B[4, k] -> C[4, k]
        U[4 * 5 + 4][col] = 1
        V[4 * 5 + k][col] = 1
        W[k * 5 + 4][col] = 1
        col += 1
        
    # Part 6: C[4, 4] = A[4, 0:4] @ B[0:4, 4]
    for j in range(4):
        # A[4, j] * B[j, 4] -> C[4, 4]
        U[4 * 5 + j][col] = 1
        V[j * 5 + 4][col] = 1
        W[4 * 5 + 4][col] = 1
        col += 1
        
    # Part 7: C[4, 4] += A[4, 4] * B[4, 4]
    U[4 * 5 + 4][col] = 1
    V[4 * 5 + 4][col] = 1
    W[4 * 5 + 4][col] = 1
    col += 1
    
    assert col == rank, f"Rank mismatch: {col} != {rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
