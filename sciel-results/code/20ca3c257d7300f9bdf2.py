import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # We apply it to 7 blocks of 2 columns each.
    # Block b corresponds to columns 2b and 2b+1.
    
    for b in range(7):
        base_col = b * 7
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Term 0: M1 = (A00 + A11) * (B00 + B11)
        # U: A00, A11
        U[0][base_col] = 1
        U[3][base_col] = 1
        # V: B00 (j=0, k=k0), B11 (j=1, k=k1)
        V[0 * p + k0][base_col] = 1
        V[1 * p + k1][base_col] = 1
        # W: C00 (k=k0, i=0), C11 (k=k1, i=1)
        W[k0 * n + 0][base_col] = 1
        W[k1 * n + 1][base_col] = 1
        
        # Term 1: M2 = (A10 + A11) * B00
        # U: A10, A11
        U[2][base_col + 1] = 1
        U[3][base_col + 1] = 1
        # V: B00 (j=0, k=k0)
        V[0 * p + k0][base_col + 1] = 1
        # W: C10 (k=k0, i=1), C11 (k=k1, i=1) with -1
        W[k0 * n + 1][base_col + 1] = 1
        W[k1 * n + 1][base_col + 1] = -1
        
        # Term 2: M3 = A00 * (B01 - B11)
        # U: A00
        U[0][base_col + 2] = 1
        # V: B01 (j=0, k=k1), B11 (j=1, k=k1) with -1
        V[0 * p + k1][base_col + 2] = 1
        V[1 * p + k1][base_col + 2] = -1
        # W: C01 (k=k1, i=0), C11 (k=k1, i=1)
        W[k1 * n + 0][base_col + 2] = 1
        W[k1 * n + 1][base_col + 2] = 1
        
        # Term 3: M4 = A11 * (B10 - B00)
        # U: A11
        U[3][base_col + 3] = 1
        # V: B10 (j=1, k=k0), B00 (j=0, k=k0) with -1
        V[1 * p + k0][base_col + 3] = 1
        V[0 * p + k0][base_col + 3] = -1
        # W: C00 (k=k0, i=0), C10 (k=k0, i=1)
        W[k0 * n + 0][base_col + 3] = 1
        W[k0 * n + 1][base_col + 3] = 1
        
        # Term 4: M5 = (A00 + A01) * B11
        # U: A00, A01
        U[0][base_col + 4] = 1
        U[1][base_col + 4] = 1
        # V: B11 (j=1, k=k1)
        V[1 * p + k1][base_col + 4] = 1
        # W: C00 (k=k0, i=0) with -1, C01 (k=k1, i=0)
        W[k0 * n + 0][base_col + 4] = -1
        W[k1 * n + 0][base_col + 4] = 1
        
        # Term 5: M6 = (A01 - A10) * (B00 + B01)
        # U: A01, A10 with -1
        U[1][base_col + 5] = 1
        U[2][base_col + 5] = -1
        # V: B00 (j=0, k=k0), B01 (j=0, k=k1)
        V[0 * p + k0][base_col + 5] = 1
        V[0 * p + k1][base_col + 5] = 1
        # W: C11 (k=k1, i=1)
        W[k1 * n + 1][base_col + 5] = 1
        
        # Term 6: M7 = (A10 - A00) * (B10 + B11)
        # U: A10, A00 with -1
        U[2][base_col + 6] = 1
        U[0][base_col + 6] = -1
        # V: B10 (j=1, k=k0), B11 (j=1, k=k1)
        V[1 * p + k0][base_col + 6] = 1
        V[1 * p + k1][base_col + 6] = 1
        # W: C00 (k=k0, i=0)
        W[k0 * n + 0][base_col + 6] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
