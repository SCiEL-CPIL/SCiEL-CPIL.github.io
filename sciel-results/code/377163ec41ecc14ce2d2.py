"""Rewritten program using Strassen's algorithm for 2x2x2 blocks to reduce rank."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # p = 11, so we have 5 pairs of columns (0,1), (2,3), (4,5), (6,7), (8,9) and one single column (10)
    # For each pair, use Strassen's algorithm (rank 7)
    # For the single column, use trivial decomposition (rank 4)
    # Total rank = 5 * 7 + 4 = 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]  # 4 x 39
    V = [[0] * rank for _ in range(m * p)]  # 22 x 39
    W = [[0] * rank for _ in range(p * n)]  # 22 x 39
    
    col = 0
    # Process 5 pairs of columns using Strassen
    for pair_idx in range(5):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        
        # Strassen's 7 products for multiplying 2x2 by 2x2 (columns k1, k2)
        # M1 = (A00 + A11)(B0k1 + B1k2)
        # M2 = (A10 + A11)B0k1
        # M3 = A00(B1k1 - B1k2)
        # M4 = A11(B0k2 - B0k1)
        # M5 = (A00 + A10)B1k1
        # M6 = (A11 - A00)(B0k1 + B0k2)
        # M7 = (A10 - A11)(B1k2 + B1k1)
        
        # M1
        U[0][col] = 1   # A00
        U[3][col] = 1   # A11
        V[0 * p + k1][col] = 1   # B0k1
        V[1 * p + k2][col] = 1   # B1k2
        W[k1 * n + 0][col] = 1   # C0,k1
        W[k1 * n + 1][col] = 1   # C1,k1
        W[k2 * n + 0][col] = 1   # C0,k2
        W[k2 * n + 1][col] = 1   # C1,k2
        col += 1
        
        # M2
        U[2][col] = 1   # A10
        U[3][col] = 1   # A11
        V[0 * p + k1][col] = 1   # B0k1
        W[k1 * n + 1][col] = 1   # C1,k1
        W[k2 * n + 1][col] = 1   # C1,k2
        col += 1
        
        # M3
        U[0][col] = 1   # A00
        V[1 * p + k1][col] = 1   # B1k1
        V[1 * p + k2][col] = -1  # B1k2
        W[k1 * n + 0][col] = 1   # C0,k1
        W[k2 * n + 0][col] = -1  # C0,k2
        col += 1
        
        # M4
        U[3][col] = 1   # A11
        V[0 * p + k2][col] = 1   # B0k2
        V[0 * p + k1][col] = -1  # B0k1
        W[k2 * n + 1][col] = 1   # C1,k2
        W[k1 * n + 1][col] = -1  # C1,k1
        col += 1
        
        # M5
        U[0][col] = 1   # A00
        U[2][col] = 1   # A10
        V[1 * p + k1][col] = 1   # B1k1
        W[k1 * n + 0][col] = 1   # C0,k1
        W[k1 * n + 1][col] = 1   # C1,k1
        col += 1
        
        # M6
        U[3][col] = 1   # A11
        U[0][col] = -1  # A00
        V[0 * p + k1][col] = 1   # B0k1
        V[0 * p + k2][col] = 1   # B0k2
        W[k1 * n + 0][col] = 1   # C0,k1
        W[k2 * n + 0][col] = 1   # C0,k2
        col += 1
        
        # M7
        U[2][col] = 1   # A10
        U[3][col] = -1  # A11
        V[1 * p + k2][col] = 1   # B1k2
        V[1 * p + k1][col] = 1   # B1k1
        W[k2 * n + 1][col] = 1   # C1,k2
        W[k2 * n + 0][col] = 1   # C0,k2
        col += 1
    
    # Process last column (k=10) with trivial decomposition (rank 4)
    k = 10
    # C0,k = A00*B0k + A01*B1k
    # C1,k = A10*B0k + A11*B1k
    
    # Term 1: A00 * B0k -> C0,k
    U[0][col] = 1
    V[0 * p + k][col] = 1
    W[k * n + 0][col] = 1
    col += 1
    
    # Term 2: A01 * B1k -> C0,k
    U[1][col] = 1
    V[1 * p + k][col] = 1
    W[k * n + 0][col] = 1
    col += 1
    
    # Term 3: A10 * B0k -> C1,k
    U[2][col] = 1
    V[0 * p + k][col] = 1
    W[k * n + 1][col] = 1
    col += 1
    
    # Term 4: A11 * B1k -> C1,k
    U[3][col] = 1
    V[1 * p + k][col] = 1
    W[k * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
