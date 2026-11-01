"""Matrix multiplication 2,2,12 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    The multiplication tensor is a direct sum of 6 independent 2x2x2 multiplication tensors 
    (grouping columns in pairs). The minimal rank for 2x2x2 is 7 (Strassen).
    Total rank = 6 * 7 = 42.
    
    Interface:
    U: shape (n*m, rank) = (4, 42)
    V: shape (m*p, rank) = (24, 42)
    W: shape (p*n, rank) = (24, 42)
    
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    t_start = time.time()
    deadline = t_start + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_pairs = p // 2  # 6
    rank = num_pairs * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen's algorithm coefficients
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Pre-compute the Strassen factors for one 2x2 block
    # U_s: (4, 7)
    # V_s: (4, 7)  # B has 2 rows, 2 cols in the block -> 4 entries
    # W_s: (4, 7)  # C has 2 rows, 2 cols in the block -> 4 entries
    
    U_s = np.zeros((4, 7), dtype=np.float64)
    V_s = np.zeros((4, 7), dtype=np.float64)
    W_s = np.zeros((4, 7), dtype=np.float64)
    
    # s=0: M1
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1  # B[0,0], B[1,1]
    W_s[0, 0] = 1; W_s[3, 0] = 1  # C[0,0], C[1,1]
    
    # s=1: M2
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1  # B[0,0]
    W_s[2, 1] = 1; W_s[3, 1] = -1  # C[1,0], C[1,1]
    
    # s=2: M3
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1  # B[0,1], B[1,1]
    W_s[1, 2] = 1; W_s[3, 2] = 1  # C[0,1], C[1,1]
    
    # s=3: M4
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1  # B[1,0], B[0,0]
    W_s[0, 3] = 1; W_s[2, 3] = 1  # C[0,0], C[1,0]
    
    # s=4: M5
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1  # B[1,1]
    W_s[0, 4] = -1; W_s[1, 4] = 1  # C[0,0], C[0,1]
    
    # s=5: M6
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1  # B[0,0], B[0,1]
    W_s[3, 5] = 1  # C[1,1]
    
    # s=6: M7
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1  # B[1,0], B[1,1]
    W_s[0, 6] = 1  # C[0,0]

    # Map the local indices (0..3) for B and C to global indices
    # B global index: j*p + k. In the block, j=0,1 and k=0,1 (local).
    # Global k = 2*t + local_k.
    # C global index: k*n + i. In the block, i=0,1 and k=0,1 (local).
    # Global k = 2*t + local_k.
    
    for t in range(num_pairs):
        if time.time() > deadline:
            break
        base = t * 7
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # U is shared across blocks, so we just add the same U_s to the slice
        U[:, base:base+7] += U_s
        
        # V mapping:
        # Local B[0,0] -> Global B[0, k0] -> Index 0*p + k0
        # Local B[0,1] -> Global B[0, k1] -> Index 0*p + k1
        # Local B[1,0] -> Global B[1, k0] -> Index 1*p + k0
        # Local B[1,1] -> Global B[1, k1] -> Index 1*p + k1
        
        V[0*p + k0, base:base+7] += V_s[0, :]
        V[0*p + k1, base:base+7] += V_s[1, :]
        V[1*p + k0, base:base+7] += V_s[2, :]
        V[1*p + k1, base:base+7] += V_s[3, :]
        
        # W mapping:
        # Local C[0,0] -> Global C[0, k0] -> Index k0*n + 0
        # Local C[0,1] -> Global C[0, k1] -> Index k1*n + 0
        # Local C[1,0] -> Global C[1, k0] -> Index k0*n + 1
        # Local C[1,1] -> Global C[1, k1] -> Index k1*n + 1
        
        W[k0*n + 0, base:base+7] += W_s[0, :]
        W[k1*n + 0, base:base+7] += W_s[1, :]
        W[k0*n + 1, base:base+7] += W_s[2, :]
        W[k1*n + 1, base:base+7] += W_s[3, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
