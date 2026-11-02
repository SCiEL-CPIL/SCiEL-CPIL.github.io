"""Rewritten program for M_{2,2,15} using Strassen's algorithm for 2x2 blocks."""

def construct():
    # Dimensions: A is 2x2, B is 2x15, C is 2x15
    n, m, p = 2, 2, 15
    
    # We group the 15 columns of B into 7 blocks of 2 columns and 1 block of 1 column.
    # Each 2-column block uses Strassen's algorithm for M_{2,2,2} (rank 7).
    # The last 1-column block uses schoolbook for M_{2,2,1} (rank 4).
    # Total rank = 7 * 7 + 4 = 53.
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # Size 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # Size 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # Size 30 x 53
    
    col_idx = 0
    
    # Process 7 blocks of 2 columns (columns 0-1, 2-3, ..., 12-13)
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen's 7 multiplications
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0][col_idx] = 1
        U[3][col_idx] = 1
        V[k0][col_idx] = 1
        V[15 + k1][col_idx] = 1
        W[2 * k0][col_idx] = 1       # C0k0
        W[2 * k1 + 1][col_idx] = 1   # C1k1
        col_idx += 1
        
        # M2 = (A10 + A11) * B0k0
        U[2][col_idx] = 1
        U[3][col_idx] = 1
        V[k0][col_idx] = 1
        W[2 * k0 + 1][col_idx] = 1   # C1k0
        W[2 * k1 + 1][col_idx] = -1  # C1k1
        col_idx += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0][col_idx] = 1
        V[k1][col_idx] = 1
        V[15 + k1][col_idx] = -1
        W[2 * k1][col_idx] = 1       # C0k1
        W[2 * k1 + 1][col_idx] = 1   # C1k1
        col_idx += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[3][col_idx] = 1
        V[15 + k0][col_idx] = 1
        V[k0][col_idx] = -1
        W[2 * k0][col_idx] = 1       # C0k0
        W[2 * k0 + 1][col_idx] = 1   # C1k0
        col_idx += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0][col_idx] = 1
        U[1][col_idx] = 1
        V[15 + k1][col_idx] = 1
        W[2 * k0][col_idx] = -1      # C0k0
        W[2 * k1][col_idx] = 1       # C0k1
        col_idx += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[2][col_idx] = 1
        U[0][col_idx] = -1
        V[k0][col_idx] = 1
        V[k1][col_idx] = 1
        W[2 * k1 + 1][col_idx] = 1   # C1k1
        col_idx += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1][col_idx] = 1
        U[3][col_idx] = -1
        V[15 + k0][col_idx] = 1
        V[15 + k1][col_idx] = 1
        W[2 * k0][col_idx] = 1       # C0k0
        col_idx += 1
        
    # Process the last single column (column 14)
    k = 14
    # Schoolbook 4 terms
    # C0k = A00 B0k + A01 B1k
    # C1k = A10 B0k + A11 B1k
    
    # Term 1: A00 * B0k -> C0k
    U[0][col_idx] = 1
    V[k][col_idx] = 1
    W[2 * k][col_idx] = 1
    col_idx += 1
    
    # Term 2: A01 * B1k -> C0k
    U[1][col_idx] = 1
    V[15 + k][col_idx] = 1
    W[2 * k][col_idx] = 1
    col_idx += 1
    
    # Term 3: A10 * B0k -> C1k
    U[2][col_idx] = 1
    V[k][col_idx] = 1
    W[2 * k + 1][col_idx] = 1
    col_idx += 1
    
    # Term 4: A11 * B1k -> C1k
    U[3][col_idx] = 1
    V[15 + k][col_idx] = 1
    W[2 * k + 1][col_idx] = 1
    col_idx += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
