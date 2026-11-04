import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2.
    Returns U, V, W of shape (4, 7) each.
    U[i*2+j, r] for i in 0..1, j in 0..1
    V[j*2+k, r] for j in 0..1, k in 0..1
    W[k*2+i, r] for k in 0..1, i in 0..1
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1: (A00+A11)(B00+B11)
    U[0*2+0, 0] = 1  # A00
    U[1*2+1, 0] = 1  # A11
    V[0*2+0, 0] = 1  # B00
    V[1*2+1, 0] = 1  # B11
    W[0*2+0, 0] = 1  # C00
    W[1*2+1, 0] = 1  # C11
    
    # M2: (A10+A11)B00
    U[1*2+0, 1] = 1  # A10
    U[1*2+1, 1] = 1  # A11
    V[0*2+0, 1] = 1  # B00
    W[1*2+0, 1] = 1  # C10
    
    # M3: A00(B01-B11)
    U[0*2+0, 2] = 1  # A00
    V[0*2+1, 2] = 1  # B01
    V[1*2+1, 2] = -1 # B11
    W[0*2+1, 2] = 1  # C01
    
    # M4: A11(B10-B00)
    U[1*2+1, 3] = 1  # A11
    V[1*2+0, 3] = 1  # B10
    V[0*2+0, 3] = -1 # B00
    W[0*2+0, 3] = 1  # C00
    W[1*2+0, 3] = 1  # C10
    
    # M5: (A00+A01)B11
    U[0*2+0, 4] = 1  # A00
    U[0*2+1, 4] = 1  # A01
    V[1*2+1, 4] = 1  # B11
    W[0*2+0, 4] = -1 # C00
    W[0*2+1, 4] = 1  # C01
    
    # M6: (A10-A00)(B00+B01)
    U[1*2+0, 5] = 1  # A10
    U[0*2+0, 5] = -1 # A00
    V[0*2+0, 5] = 1  # B00
    V[0*2+1, 5] = 1  # B01
    W[1*2+1, 5] = 1  # C11
    
    # M7: (A01-A11)(B10+B11)
    U[0*2+1, 6] = 1  # A01
    U[1*2+1, 6] = -1 # A11
    V[1*2+0, 6] = 1  # B10
    V[1*2+1, 6] = 1  # B11
    W[0*2+0, 6] = 1  # C00
    W[1*2+1, 6] = 1  # C11
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify if U, V, W exactly decompose the matrix multiplication tensor."""
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(R):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if s != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    deadline = time.time() + 140  # 10s safety margin
    
    # Block decomposition: split m=5 into groups of 2, 2, 1
    # Group 1: j=0,1 (2 middle indices) -> 2x2 x 2x6 subproblem
    # Group 2: j=2,3 (2 middle indices) -> 2x2 x 2x6 subproblem
    # Group 3: j=4 (1 middle index) -> 2x1 x 1x6 subproblem
    
    # For each 2x2 x 2x6 subproblem:
    #   A is 2x2, B is 2x6. Split B into three 2x2 blocks.
    #   Each 2x2 x 2x2 uses Strassen (rank 7).
    #   Total rank per 2x2 x 2x6 = 3 * 7 = 21.
    
    # For 2x1 x 1x6 subproblem:
    #   A is 2x1, B is 1x6. 12 independent products, rank 12.
    
    # Total rank = 21 + 21 + 12 = 54.
    
    rank_total = 54
    U = np.zeros((n * m, rank_total), dtype=int)
    V = np.zeros((m * p, rank_total), dtype=int)
    W = np.zeros((p * n, rank_total), dtype=int)
    
    # Get Strassen decomposition
    U_s, V_s, W_s = _strassen_2x2x2()
    
    col = 0
    # Group 1: j=0,1
    # A indices: i=0,1; j=0,1 -> A[i, j] at U[i*m + j, :]
    # B indices: j=0,1; k=0..5 -> B[j, k] at V[j*p + k, :]
    # C indices: i=0,1; k=0..5 -> C[i, k] at W[k*n + i, :]
    
    # For B, split k into 0,1 / 2,3 / 4,5
    for k_group in range(3):
        k0 = k_group * 2
        k1 = k_group * 2 + 1
        for r in range(7):
            # A part: U_s has entries for A[i, j] where i in 0..1, j in 0..1
            # Map to global U: i_global = i, j_global = j (since j=0,1 for group 1)
            for i in range(2):
                for j in range(2):
                    if U_s[i*2+j, r] != 0:
                        U[i*m + j, col] = U_s[i*2+j, r]
            
            # B part: V_s has entries for B[j, k] where j in 0..1, k in 0..1
            # Map to global V: j_global = j (0,1), k_global = k0 + k
            for j in range(2):
                for k_local in range(2):
                    if V_s[j*2+k_local, r] != 0:
                        V[j*p + (k0 + k_local), col] = V_s[j*2+k_local, r]
            
            # C part: W_s has entries for C[i, k] where i in 0..1, k in 0..1
            # Map to global W: k_global = k0 + k, i_global = i
            for k_local in range(2):
                for i in range(2):
                    if W_s[k_local*2+i, r] != 0:
                        W[(k0 + k_local)*n + i, col] = W_s[k_local*2+i, r]
            
            col += 1
    
    # Group 2: j=2,3
    for k_group in range(3):
        k0 = k_group * 2
        k1 = k_group * 2 + 1
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    if U_s[i*2+j, r] != 0:
                        U[i*m + (j+2), col] = U_s[i*2+j, r]
            
            for j in range(2):
                for k_local in range(2):
                    if V_s[j*2+k_local, r] != 0:
                        V[(j+2)*p + (k0 + k_local), col] = V_s[j*2+k_local, r]
            
            for k_local in range(2):
                for i in range(2):
                    if W_s[k_local*2+i, r] != 0:
                        W[(k0 + k_local)*n + i, col] = W_s[k_local*2+i, r]
            
            col += 1
    
    # Group 3: j=4 (single middle index)
    # A is 2x1: A[i, 4] for i=0,1
    # B is 1x6: B[4, k] for k=0..5
    # C[i, k] = A[i, 4] * B[4, k]
    # 12 independent products
    for i in range(2):
        for k in range(6):
            U[i*m + 4, col] = 1
            V[4*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
