import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    rank = 62
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Schoolbook for indices not in the 2x2x2 core
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue # Skip these, handled by Strassen
                U[i * m + j, col_idx] = 1
                V[j * p + k, col_idx] = 1
                W[k * n + i, col_idx] = 1
                col_idx += 1
                if col_idx >= rank:
                    break
            if col_idx >= rank:
                break
        if col_idx >= rank:
            break
            
    # Now add the 7 Strassen columns for the 2x2x2 core
    # Indices for U: i*3+j
    # Indices for V: j*7+k
    # Indices for W: k*3+i
    
    # Strassen definitions
    # M1: (A00+A11)(B00+B11)
    U[0, col_idx] = 1; U[4, col_idx] = 1
    V[0, col_idx] = 1; V[8, col_idx] = 1
    W[0, col_idx] = 1; W[4, col_idx] = 1
    col_idx += 1
    
    # M2: (A10+A11)B00
    U[4, col_idx] = 1; U[5, col_idx] = 1 # A10 is i=1,j=0 -> 1*3+0=3? No.
    # A[i,j]. i=1, j=0 -> index 3.
    # A[i,j]. i=1, j=1 -> index 4.
    # So M2 uses A[1,0] and A[1,1].
    U[3, col_idx] = 1; U[4, col_idx] = 1
    V[0, col_idx] = 1
    W[3, col_idx] = 1 # C10
    col_idx += 1
    
    # M3: A00(B01-B11)
    U[0, col_idx] = 1
    V[1, col_idx] = 1; V[8, col_idx] = -1
    W[1, col_idx] = 1 # C01
    col_idx += 1
    
    # M4: A11(B10-B00)
    U[4, col_idx] = 1
    V[7, col_idx] = 1; V[0, col_idx] = -1
    W[3, col_idx] = 1; W[0, col_idx] = 1 # C10, C00
    col_idx += 1
    
    # M5: (A00+A01)B11
    U[0, col_idx] = 1; U[1, col_idx] = 1
    V[8, col_idx] = 1
    W[0, col_idx] = -1; W[1, col_idx] = 1 # -C00, +C01
    col_idx += 1
    
    # M6: (A10-A00)(B00+B01)
    U[3, col_idx] = 1; U[0, col_idx] = -1
    V[0, col_idx] = 1; V[1, col_idx] = 1
    W[4, col_idx] = 1 # C11
    col_idx += 1
    
    # M7: (A01-A11)(B10+B11)
    U[1, col_idx] = 1; U[4, col_idx] = -1
    V[7, col_idx] = 1; V[8, col_idx] = 1
    W[0, col_idx] = 1 # C00
    col_idx += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
