import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 17
    target_rank = 17
    
    # Strassen factors for 2x2x2
    # U_s: 4x7 (i in 0,1; j in 0,1)
    # V_s: 4x7 (j in 0,1; k in 0,1)
    # W_s: 4x7 (i in 0,1; k in 0,1) - Note: W_s[i,k] corresponds to C[i,k]
    
    # Standard Strassen
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M2 + M3 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 + M4 - M5 + M7
    
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1
        [1, 1, 0, 1, 0, 1, -1], # i=1, j=0
        [1, 0, -1, 0, 0, 0, 1]  # i=1, j=1
    ], dtype=float)
    
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0
        [0, 0, 1, 1, 0, 0, 1],  # j=0, k=1
        [1, 0, -1, 0, 1, 0, 1], # j=1, k=0
        [1, 0, 0, 0, 1, 0, 0]   # j=1, k=1
    ], dtype=float)
    
    W_s = np.array([
        [1, -1, 1, 0, 0, 1, 0], # i=0, k=0 (C00)
        [0, 0, 1, 0, 1, 0, 0],  # i=0, k=1 (C01)
        [0, 1, 0, 1, 0, 0, 0],  # i=1, k=0 (C10)
        [1, 0, 0, 1, -1, 0, 1]  # i=1, k=1 (C11)
    ], dtype=float)
    
    # Initialize full matrices
    U = np.zeros((n*m, target_rank), dtype=float)
    V = np.zeros((m*p, target_rank), dtype=float)
    W = np.zeros((p*n, target_rank), dtype=float)
    
    # Embed Strassen in j in {0,1}, k in {0,1}
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    for i in range(2):
        for j in range(2):
            U[i*m + j, 0:7] = U_s[i*2 + j, :]
            
    for j in range(2):
        for k in range(2):
            V[j*p + k, 0:7] = V_s[j*2 + k, :]
            
    for i in range(2):
        for k in range(2):
            # W_s is indexed by i*2 + k
            # W global is indexed by k*n + i
            W[k*n + i, 0:7] = W_s[i*2 + k, :]
            
    # Add schoolbook for remaining 10 terms
    # Terms where j=2 OR k=2 (excluding j<2 and k<2)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j < 2 and k < 2:
                    continue # Already covered by Strassen
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    if col != target_rank:
        # Fallback if counting mismatch
        return _schoolbook()
        
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return _schoolbook()

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p):
    # Check sum_c U_ij,c V_jk,c W_ik,c == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
