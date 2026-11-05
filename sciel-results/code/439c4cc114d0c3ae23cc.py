import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 88
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen blocks for i in {0, 1}
    # j pairs: (0, 1), (2, 3)
    # k pairs: (0, 1), (2, 3), (4, 5), (6, 7)
    i_base = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for j_base in [0, 2]:
        for k_base in [0, 2, 4, 6]:
            # 7 Strassen terms
            # M0 = (A00+A11)(B00+B11)
            U[i_base*m + j_base, col] = 1
            U[(i_base+1)*m + j_base+1, col] = 1
            V[j_base*p + k_base, col] = 1
            V[j_base+1*p + k_base+1, col] = 1
            W[k_base*n + i_base, col] = 1
            W[k_base+1*n + i_base+1, col] = 1
            col += 1
            
            # M1 = (A10+A11)B00
            U[i_base*m + j_base+1, col] = 1
            U[(i_base+1)*m + j_base+1, col] = 1
            V[j_base*p + k_base, col] = 1
            W[k_base*n + i_base+1, col] = 1
            col += 1
            
            # M2 = A00(B01-B11)
            U[i_base*m + j_base, col] = 1
            V[j_base*p + k_base+1, col] = 1
            V[j_base+1*p + k_base+1, col] = -1
            W[k_base+1*n + i_base, col] = 1
            col += 1
            
            # M3 = A11(B10-B00)
            U[(i_base+1)*m + j_base+1, col] = 1
            V[j_base+1*p + k_base, col] = 1
            V[j_base*p + k_base, col] = -1
            W[k_base*n + i_base+1, col] = 1
            col += 1
            
            # M4 = (A00+A01)B11
            U[i_base*m + j_base, col] = 1
            U[i_base*m + j_base+1, col] = 1
            V[j_base+1*p + k_base+1, col] = 1
            W[k_base+1*n + i_base, col] = 1
            col += 1
            
            # M5 = (A10-A00)(B00+B01)
            U[i_base*m + j_base+1, col] = 1
            U[i_base*m + j_base, col] = -1
            V[j_base*p + k_base, col] = 1
            V[j_base*p + k_base+1, col] = 1
            W[k_base+1*n + i_base+1, col] = 1
            col += 1
            
            # M6 = (A01-A11)(B10+B11)
            U[i_base*m + j_base+1, col] = 1
            U[(i_base+1)*m + j_base+1, col] = -1
            V[j_base+1*p + k_base, col] = 1
            V[j_base+1*p + k_base+1, col] = 1
            W[k_base*n + i_base, col] = 1
            col += 1

    # Schoolbook for i = 2
    i_base = 2
    for j in range(m):
        for k in range(p):
            U[i_base*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + i_base, col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
