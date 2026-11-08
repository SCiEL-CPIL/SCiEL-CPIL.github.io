"""Rewritten program: Strassen block decomposition for 2x4 * 4x8 matrix multiplication."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    rank = 56
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Split the 4 intermediate indices into 2 pairs
    j_pairs = [(0, 1), (2, 3)]
    # Split the 8 output columns into 4 pairs
    k_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    col = 0
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            # Strassen M1: (A00 + A11) * (B00 + B11)
            U[0 * m + j1][col] = 1
            U[1 * m + j2][col] = 1
            V[j1 * p + k1][col] = 1
            V[j2 * p + k2][col] = 1
            W[k1 * n + 0][col] = 1
            W[k2 * n + 1][col] = 1
            col += 1
            
            # Strassen M2: (A10 + A11) * B00
            U[1 * m + j1][col] = 1
            U[1 * m + j2][col] = 1
            V[j1 * p + k1][col] = 1
            W[k1 * n + 1][col] = 1
            W[k2 * n + 1][col] = -1
            col += 1
            
            # Strassen M3: A00 * (B01 - B11)
            U[0 * m + j1][col] = 1
            V[j1 * p + k2][col] = 1
            V[j2 * p + k2][col] = -1
            W[k2 * n + 0][col] = 1
            W[k2 * n + 1][col] = 1
            col += 1
            
            # Strassen M4: A11 * (B10 - B00)
            U[1 * m + j2][col] = 1
            V[j2 * p + k1][col] = 1
            V[j1 * p + k1][col] = -1
            W[k1 * n + 0][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1
            
            # Strassen M5: (A00 + A01) * B11
            U[0 * m + j1][col] = 1
            U[0 * m + j2][col] = 1
            V[j2 * p + k2][col] = 1
            W[k1 * n + 0][col] = -1
            W[k2 * n + 0][col] = 1
            col += 1
            
            # Strassen M6: (A10 - A00) * (B00 + B01)
            U[1 * m + j1][col] = 1
            U[0 * m + j1][col] = -1
            V[j1 * p + k1][col] = 1
            V[j1 * p + k2][col] = 1
            W[k2 * n + 1][col] = 1
            col += 1
            
            # Strassen M7: (A01 - A11) * (B10 + B11)
            U[0 * m + j2][col] = 1
            U[1 * m + j2][col] = -1
            V[j2 * p + k1][col] = 1
            V[j2 * p + k2][col] = 1
            W[k1 * n + 0][col] = -1
            col += 1
    
    assert col == rank, f"Column index {col} != rank {rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
