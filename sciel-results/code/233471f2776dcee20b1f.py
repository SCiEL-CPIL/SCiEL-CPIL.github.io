import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U is 4x7, V is 4x7, W is 4x7.
    Indices:
    U: i*2 + j
    V: j*2 + k
    W: k*2 + i
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00, C11 get M1
    
    # M2 = (A10 + A11) B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[3, 1] = 1; W[2, 1] = -1 # C11 - M2, C10 + M2
    
    # M3 = A00 (B10 - B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C01 + M3, C11 + M3
    
    # M4 = A11 (B01 - B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C00 + M4, C10 + M4
    
    # M5 = (A00 + A01) B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1 # C01 + M5, C00 - M5
    
    # M6 = (A10 - A00)(B00 + B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1 # C11 + M6
    
    # M7 = (A01 - A11)(B01 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00 + M7
    
    return U, V, W

def _schoolbook_126():
    """
    Returns U, V, W for 1x2x6 matrix multiplication using schoolbook.
    U is 2x12, V is 12x12, W is 6x12.
    Indices:
    U: 0*2 + j = j (j=0,1)
    V: j*6 + k
    W: k*1 + 0 = k
    """
    rank = 12
    U = np.zeros((2, rank))
    V = np.zeros((12, rank))
    W = np.zeros((6, rank))
    
    col = 0
    for j in range(2):
        for k in range(6):
            U[j, col] = 1
            V[j*6 + k, col] = 1
            W[k, col] = 1
            col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=6
    
    # We will build U, V, W as lists of columns, then stack.
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Block 1: M_2,2,6 at i in {0,1}, j in {0,1}, k in {0..5}
    # Split k into 3 pairs: (0,1), (2,3), (4,5)
    for k_start in [0, 2, 4]:
        Us, Vs, Ws = _strassen_222()
        # Map local indices to global
        # Local i: 0,1 -> Global i: 0,1
        # Local j: 0,1 -> Global j: 0,1
        # Local k: 0,1 -> Global k: k_start, k_start+1
        
        U_map = np.zeros((12, 7))
        V_map = np.zeros((24, 7))
        W_map = np.zeros((18, 7))
        
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    val = Us[li*2 + lj, r]
                    if val != 0:
                        gi = li
                        gj = lj
                        U_map[gi*m + gj, r] += val
                        
            for lj in range(2):
                for lk in range(2):
                    val = Vs[lj*2 + lk, r]
                    if val != 0:
                        gj = lj
                        gk = k_start + lk
                        V_map[gj*p + gk, r] += val
                        
            for lk in range(2):
                for li in range(2):
                    val = Ws[lk*2 + li, r]
                    if val != 0:
                        gk = k_start + lk
                        gi = li
                        W_map[gk*n + gi, r] += val
                        
        U_cols.append(U_map)
        V_cols.append(V_map)
        W_cols.append(W_map)
        
    # Block 2: M_2,2,6 at i in {0,1}, j in {2,3}, k in {0..5}
    for k_start in [0, 2, 4]:
        Us, Vs, Ws = _strassen_222()
        U_map = np.zeros((12, 7))
        V_map = np.zeros((24, 7))
        W_map = np.zeros((18, 7))
        
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    val = Us[li*2 + lj, r]
                    if val != 0:
                        gi = li
                        gj = 2 + lj
                        U_map[gi*m + gj, r] += val
                        
            for lj in range(2):
                for lk in range(2):
                    val = Vs[lj*2 + lk, r]
                    if val != 0:
                        gj = 2 + lj
                        gk = k_start + lk
                        V_map[gj*p + gk, r] += val
                        
            for lk in range(2):
                for li in range(2):
                    val = Ws[lk*2 + li, r]
                    if val != 0:
                        gk = k_start + lk
                        gi = li
                        W_map[gk*n + gi, r] += val
                        
        U_cols.append(U_map)
        V_cols.append(V_map)
        W_cols.append(W_map)
        
    # Block 3: M_1,2,6 at i=2, j in {0,1}, k in {0..5}
    Us, Vs, Ws = _schoolbook_126()
    U_map = np.zeros((12, 12))
    V_map = np.zeros((24, 12))
    W_map = np.zeros((18, 12))
    
    for r in range(12):
        for lj in range(2):
            val = Us[lj, r]
            if val != 0:
                gi = 2
                gj = lj
                U_map[gi*m + gj, r] += val
                
        for lj in range(2):
            for lk in range(6):
                val = Vs[lj*6 + lk, r]
                if val != 0:
                    gj = lj
                    gk = lk
                    V_map[gj*p + gk, r] += val
                    
        for lk in range(6):
            val = Ws[lk, r]
            if val != 0:
                gk = lk
                gi = 2
                W_map[gk*n + gi, r] += val
                
    U_cols.append(U_map)
    V_cols.append(V_map)
    W_cols.append(W_map)
    
    # Block 4: M_1,2,6 at i=2, j in {2,3}, k in {0..5}
    Us, Vs, Ws = _schoolbook_126()
    U_map = np.zeros((12, 12))
    V_map = np.zeros((24, 12))
    W_map = np.zeros((18, 12))
    
    for r in range(12):
        for lj in range(2):
            val = Us[lj, r]
            if val != 0:
                gi = 2
                gj = 2 + lj
                U_map[gi*m + gj, r] += val
                
        for lj in range(2):
            for lk in range(6):
                val = Vs[lj*6 + lk, r]
                if val != 0:
                    gj = 2 + lj
                    gk = lk
                    V_map[gj*p + gk, r] += val
                    
        for lk in range(6):
            val = Ws[lk, r]
            if val != 0:
                gk = lk
                gi = 2
                W_map[gk*n + gi, r] += val
                
    U_cols.append(U_map)
    V_cols.append(V_map)
    W_cols.append(W_map)
    
    U = np.hstack(U_cols)
    V = np.hstack(V_cols)
    W = np.hstack(W_cols)
    
    # Convert to lists of lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
