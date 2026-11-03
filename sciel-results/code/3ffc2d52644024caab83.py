import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4x4 matrix multiplication.
    
    Uses the Kronecker product of Strassen's rank-7 decomposition for 2x2x2.
    This gives rank 7*7 = 49, which is a known valid decomposition.
    
    The matrix multiplication tensor M_{n,m,p} has entries:
    M[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    
    A rank-r decomposition is: M = sum_c u_c ⊗ v_c ⊗ w_c
    where u_c has length n*m, v_c has length m*p, w_c has length p*n.
    """
    # Strassen decomposition for 2x2x2
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # The 7 Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1]    # A11
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 0, 1, 0, 1, 0],    # B01
        [0, 0, 1, 0, 0, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1]    # B11
    ]
    W2 = [
        [1, 0, 0, 0, 0, 0, 0],    # C00
        [0, 0, 0, 0, 1, 0, 0],    # C01
        [0, 1, 1, 0, 0, 1, 0],    # C10
        [1, 0, 0, 0, 0, 0, 1]     # C11
    ]
    
    # Verify Strassen decomposition: for all i,j,k in {0,1},
    # sum_c U2[i*2+j][c] * V2[j*2+k][c] * W2[k*2+i][c] == 1
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j][c] * V2[j*2+k][c] * W2[k*2+i][c]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        # Construct rank-49 decomposition for 4x4x4
        n, m, p = 4, 4, 4
        rank = 49
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Kronecker product of the two 2x2x2 decompositions
        # The 4x4x4 tensor is the tensor product of two 2x2x2 tensors
        # Index mapping: i = i1*2 + i2, j = j1*2 + j2, k = k1*2 + k2
        # U index: i*4 + j = (i1*2+i2)*4 + (j1*2+j2)
        # V index: j*4 + k = (j1*2+j2)*4 + (k1*2+k2)
        # W index: k*4 + i = (k1*2+k2)*4 + (i1*2+i2)
        
        for c1 in range(7):
            for c2 in range(7):
                c = c1 * 7 + c2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                U[i * 4 + j][c] = U2[i1 * 2 + j1][c1] * U2[i2 * 2 + j2][c2]
                                
                                for k1 in range(2):
                                    for k2 in range(2):
                                        k = k1 * 2 + k2
                                        V[j * 4 + k][c] = V2[j1 * 2 + k1][c1] * V2[j2 * 2 + k2][c2]
                                        W[k * 4 + i][c] = W2[k1 * 2 + i1][c1] * W2[k2 * 2 + i2][c2]
        
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook rank-64
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
