import numpy as np

def construct():
    n, m, p = 3, 4, 4
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen helper for 2x2x2 block
    # local i, j, k in {0,1}
    # global offsets gi, gj, gk
    def add_strassen(gi, gj, gk):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[gi*m + gj + 0, col] += 1
        U[gi+1*m + gj + 1, col] += 1
        V[gj*p + gk + 0, col] += 1
        V[gj+1*p + gk + 1, col] += 1
        W[(gk+0)*n + gi + 0, col] += 1
        W[(gk+1)*n + gi + 1, col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[gi+1*m + gj + 0, col] += 1
        U[gi+1*m + gj + 1, col] += 1
        V[gj*p + gk + 0, col] += 1
        W[(gk+0)*n + gi + 1, col] += 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[gi*m + gj + 0, col] += 1
        V[gj*p + gk + 1, col] += 1
        V[gj+1*p + gk + 1, col] -= 1
        W[(gk+1)*n + gi + 0, col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[gi+1*m + gj + 1, col] += 1
        V[gj+1*p + gk + 0, col] += 1
        V[gj*p + gk + 0, col] -= 1
        W[(gk+0)*n + gi + 1, col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[gi*m + gj + 0, col] += 1
        U[gi*m + gj + 1, col] += 1
        V[gj+1*p + gk + 1, col] += 1
        W[(gk+1)*n + gi + 0, col] -= 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[gi+1*m + gj + 0, col] += 1
        U[gi*m + gj + 0, col] -= 1
        V[gj*p + gk + 0, col] += 1
        V[gj*p + gk + 1, col] += 1
        W[(gk+1)*n + gi + 1, col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[gi*m + gj + 1, col] += 1
        U[gi+1*m + gj + 1, col] -= 1
        V[gj+1*p + gk + 0, col] += 1
        V[gj+1*p + gk + 1, col] += 1
        W[(gk+0)*n + gi + 0, col] += 1
        col += 1

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen(0, 0, 0)
    
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    add_strassen(0, 2, 0)
    
    # Block 3: i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen(0, 0, 2)
    
    # Block 4: i in {0,1}, j in {2,3}, k in {2,3}
    add_strassen(0, 2, 2)
    
    # Block 5: i=2, j in {0,1,2,3}, k in {0,1}
    # C_2,0 = sum A_2,j B_j,0
    for j in range(4):
        U[2*m + j, col] += 1
        V[j*p + 0, col] += 1
        W[0*n + 2, col] += 1
        col += 1
        
    # C_2,1 = sum A_2,j B_j,1
    for j in range(4):
        U[2*m + j, col] += 1
        V[j*p + 1, col] += 1
        W[1*n + 2, col] += 1
        col += 1

    # Block 6: i=2, j in {0,1,2,3}, k in {2,3}
    # C_2,2 = sum A_2,j B_j,2
    for j in range(4):
        U[2*m + j, col] += 1
        V[j*p + 2, col] += 1
        W[2*n + 2, col] += 1
        col += 1
        
    # C_2,3 = sum A_2,j B_j,3
    for j in range(4):
        U[2*m + j, col] += 1
        V[j*p + 3, col] += 1
        W[3*n + 2, col] += 1
        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
