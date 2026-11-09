import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for a rank-110 decomposition of 5x5x5 matrix multiplication.
    Uses Strassen's algorithm for the 4x4 sub-block (rank 49) and schoolbook for the remaining 61 terms.
    U: (25, 110)
    V: (25, 110)
    W: (25, 110)
    """
    n, m, p = DIMENSIONS
    rank = 110
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # --- Part 1: Strassen for 4x4 sub-block ---
    # Define 2x2 Strassen factors
    # U2, V2 are 4x7, W2 is 4x7
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1  # C00, C11
    
    # r=1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1 # C10, C11
    
    # r=2: M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1  # C01, C11
    
    # r=3: M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1  # C00, C10
    
    # r=4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # C00, C01
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1                 # C11
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1                 # C00
    
    # Compute 4x4 factors via Kronecker product
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Place into global matrices
    # Indices 0-15 correspond to the 4x4 sub-block (rows 0-3, cols 0-3)
    U[0:16, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    # --- Part 2: Schoolbook for remaining terms ---
    col = 49
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if all indices are in the 4x4 sub-block
                if i < 4 and j < 4 and k < 4:
                    continue
                
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
