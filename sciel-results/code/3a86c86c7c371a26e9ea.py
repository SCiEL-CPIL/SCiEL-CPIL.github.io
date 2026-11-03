import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # We use a rank 28 decomposition based on Strassen for 2x2x2 blocks.
    # The indices are i in [0,1], j in [0,3], k in [0,3].
    # We split j into [0,1] and [2,3].
    # For each j-group, we split k into [0,1] and [2,3].
    # This gives 4 independent 2x2x2 multiplications.
    # Each has rank 7 (Strassen). Total rank 28.
    
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen coefficients for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U (A) coefficients for M1..M7
    # A11 is (0,0), A12 is (0,1), A21 is (1,0), A22 is (1,1)
    # M1: A11(1), A22(1)
    # M2: A21(1), A22(1)
    # M3: A11(1)
    # M4: A22(1)
    # M5: A11(1), A12(1)
    # M6: A21(1), A11(-1)
    # M7: A12(1), A22(-1)
    
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0 (A11)
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1 (A12)
        [0, 1, 0, 0, 0, 1, 0],  # i=1, j=0 (A21)
        [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1 (A22)
    ]
    
    # V (B) coefficients for M1..M7
    # B11 is (0,0), B12 is (0,1), B21 is (1,0), B22 is (1,1)
    # M1: B11(1), B22(1)
    # M2: B11(1)
    # M3: B12(1), B22(-1)
    # M4: B21(1), B11(-1)
    # M5: B22(1)
    # M6: B11(1), B12(1)
    # M7: B21(1), B22(1)
    
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0 (B11)
        [0, 0, 1, 0, 0, 1, 0],  # j=0, k=1 (B12)
        [0, 0, 0, 1, 0, 0, 1],  # j=1, k=0 (B21)
        [1, 0, -1, 0, 1, 0, 1]  # j=1, k=1 (B22)
    ]
    
    # W (C) coefficients for M1..M7
    # C11 is (0,0), C12 is (0,1), C21 is (1,0), C22 is (1,1)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],  # k=0, i=0 (C11)
        [0, 0, 1, 0, 1, 0, 0],   # k=1, i=0 (C12)
        [0, 1, 0, 1, 0, 0, 0],   # k=0, i=1 (C21)
        [1, -1, 1, 0, 0, 1, 0]   # k=1, i=1 (C22)
    ]
    
    # Map local indices (li, lj, lk) to global indices (i, j, k)
    # We have 4 blocks:
    # Block 0: j in [0,1], k in [0,1]
    # Block 1: j in [0,1], k in [2,3]
    # Block 2: j in [2,3], k in [0,1]
    # Block 3: j in [2,3], k in [2,3]
    
    col_offset = 0
    for block in range(4):
        j_base = (block // 2) * 2
        k_base = (block % 2) * 2
        
        for c in range(7):
            col = col_offset + c
            
            # Update U
            for li in range(2):
                for lj in range(2):
                    i_global = li
                    j_global = j_base + lj
                    U[i_global*m + j_global, col] = U_strassen[li*2 + lj][c]
            
            # Update V
            for lj in range(2):
                for lk in range(2):
                    j_global = j_base + lj
                    k_global = k_base + lk
                    V[j_global*p + k_global, col] = V_strassen[lj*2 + lk][c]
            
            # Update W
            for lk in range(2):
                for li in range(2):
                    k_global = k_base + lk
                    i_global = li
                    W[k_global*n + i_global, col] = W_strassen[lk*2 + li][c]
                    
            col_offset += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
