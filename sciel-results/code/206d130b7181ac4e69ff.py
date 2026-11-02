import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:, 0] (2x1) * B[0, :] (1x8) -> 2x8
    # 16 schoolbook multiplications
    for i in range(2):
        for k in range(8):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x8) -> 2x8
    # Split B[1:3, :] into 4 blocks of 2 columns each
    # Apply Strassen to each 2x2 x 2x2 block
    # A2 local indices: a00=A[0,1]=idx1, a01=A[0,2]=idx2, a10=A[1,1]=idx4, a11=A[1,2]=idx5
    # For each block b (0..3), B block columns are b*2, b*2+1
    # B local: b00=B[1,b*2]=idx(1*8+b*2), b01=B[1,b*2+1]=idx(1*8+b*2+1)
    #          b10=B[2,b*2]=idx(2*8+b*2), b11=B[2,b*2+1]=idx(2*8+b*2+1)
    # C output: c00=C[0,b*2]=W[(b*2)*2+0], c01=C[0,b*2+1]=W[(b*2+1)*2+0]
    #           c10=C[1,b*2]=W[(b*2)*2+1], c11=C[1,b*2+1]=W[(b*2+1)*2+1]
    
    for block in range(4):
        b00 = 1 * p + block * 2
        b01 = 1 * p + block * 2 + 1
        b10 = 2 * p + block * 2
        b11 = 2 * p + block * 2 + 1
        
        c00 = (block * 2) * n + 0
        c01 = (block * 2 + 1) * n + 0
        c10 = (block * 2) * n + 1
        c11 = (block * 2 + 1) * n + 1
        
        # Strassen step 1: (a00+a11)(b00+b11)
        # Contributes: +c00, +c11
        U[1][col] = 1   # a00
        U[5][col] = 1   # a11
        V[b00][col] = 1
        V[b11][col] = 1
        W[c00][col] = 1
        W[c11][col] = 1
        col += 1
        
        # Strassen step 2: (a10+a11)b00
        # Contributes: +c10, -c11
        U[4][col] = 1   # a10
        U[5][col] = 1   # a11
        V[b00][col] = 1
        W[c10][col] = 1
        W[c11][col] = -1
        col += 1
        
        # Strassen step 3: a00(b01-b11)
        # Contributes: +c01, +c11
        U[1][col] = 1   # a00
        V[b01][col] = 1
        V[b11][col] = -1
        W[c01][col] = 1
        W[c11][col] = 1
        col += 1
        
        # Strassen step 4: a11(b10-b00)
        # Contributes: +c00, +c10
        U[5][col] = 1   # a11
        V[b10][col] = 1
        V[b00][col] = -1
        W[c00][col] = 1
        W[c10][col] = 1
        col += 1
        
        # Strassen step 5: (a00+a01)b11
        # Contributes: -c00, +c01
        U[1][col] = 1   # a00
        U[2][col] = 1   # a01
        V[b11][col] = 1
        W[c00][col] = -1
        W[c01][col] = 1
        col += 1
        
        # Strassen step 6: (a10-a00)(b00+b01)
        # Contributes: +c11
        U[4][col] = 1   # a10
        U[1][col] = -1  # a00
        V[b00][col] = 1
        V[b01][col] = 1
        W[c11][col] = 1
        col += 1
        
        # Strassen step 7: (a01-a11)(b10+b11)
        # Contributes: +c00
        U[2][col] = 1   # a01
        U[5][col] = -1  # a11
        V[b10][col] = 1
        V[b11][col] = 1
        W[c00][col] = 1
        col += 1
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)
    
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
    rank_fallback = n * m * p
    U_f = [[0] * rank_fallback for _ in range(n * m)]
    V_f = [[0] * rank_fallback for _ in range(m * p)]
    W_f = [[0] * rank_fallback for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i * m + j][col] = 1
                V_f[j * p + k][col] = 1
                W_f[k * n + i][col] = 1
                col += 1
    return {"U": U_f, "V": V_f, "W": W_f}
# EVOLVE-BLOCK-END
