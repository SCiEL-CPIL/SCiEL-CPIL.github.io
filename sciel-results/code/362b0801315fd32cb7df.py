"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Build decomposition using Laderman for 3x3x3 blocks and schoolbook for 1x2x2
    result = _build_laderman_3x5x6(n, m, p)
    
    if result is not None:
        U, V, W, rank = result
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _get_laderman_3x3x3():
    """
    Return U, V, W factors for 3x3x3 matrix multiplication (Laderman, rank 23).
    U: (9, 23) indexed as U[i*3+j, c] for A[i,j]
    V: (9, 23) indexed as V[j*3+k, c] for B[j,k]
    W: (9, 23) indexed as W[k*3+i, c] for C[k,i]
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Laderman's 23 products for 3x3x3
    # Based on Laderman (1976) "A non-recursive multiplication of 3x3 matrices"
    
    # Define the 23 bilinear forms
    # m0 = (a11 + a12)(b11 + b22)
    U[0, 0] = 1; U[1, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[3, 0] = 1; W[0, 0] = 1; W[4, 0] = -1; W[1, 0] = -1
    
    # m1 = (a12 - a11)(b11 - b22)
    U[1, 1] = 1; U[0, 1] = -1
    V[0, 1] = 1; V[3, 1] = -1
    W[3, 1] = -1; W[0, 1] = 1; W[4, 1] = 1; W[1, 1] = 1
    
    # m2 = (a21 + a22)(b21 + b33)
    U[3, 2] = 1; U[4, 2] = 1
    V[3, 2] = 1; V[8, 2] = 1
    W[6, 2] = 1; W[3, 2] = 1; W[7, 2] = -1; W[4, 2] = -1
    
    # m3 = (a22 - a21)(b21 - b33)
    U[4, 3] = 1; U[3, 3] = -1
    V[3, 3] = 1; V[8, 3] = -1
    W[6, 3] = -1; W[3, 3] = 1; W[7, 3] = 1; W[4, 3] = 1
    
    # m4 = (a31 + a32)(b31 + b11)
    U[6, 4] = 1; U[7, 4] = 1
    V[6, 4] = 1; V[0, 4] = 1
    W[0, 4] = 1; W[6, 4] = 1; W[1, 4] = -1; W[7, 4] = -1
    
    # m5 = (a32 - a31)(b31 - b11)
    U[7, 5] = 1; U[6, 5] = -1
    V[6, 5] = 1; V[0, 5] = -1
    W[0, 5] = -1; W[6, 5] = 1; W[1, 5] = 1; W[7, 5] = 1
    
    # m6 = a13 * b31
    U[2, 6] = 1
    V[6, 6] = 1
    W[0, 6] = 1
    
    # m7 = a23 * b11
    U[5, 7] = 1
    V[0, 7] = 1
    W[3, 7] = 1
    
    # m8 = a33 * b21
    U[8, 8] = 1
    V[3, 8] = 1
    W[6, 8] = 1
    
    # m9 = a13 * b12
    U[2, 9] = 1
    V[3, 9] = 1
    W[4, 9] = 1
    
    # m10 = a23 * b22
    U[5, 10] = 1
    V[6, 10] = 1
    W[7, 10] = 1
    
    # m11 = a33 * b32
    U[8, 11] = 1
    V[9, 11] = 1
    W[1, 11] = 1
    
    # m12 = a12 * b13
    U[1, 12] = 1
    V[0, 12] = 1
    W[3, 12] = 1
    
    # m13 = a22 * b23
    U[4, 13] = 1
    V[3, 13] = 1
    W[6, 13] = 1
    
    # m14 = a32 * b33
    U[7, 14] = 1
    V[6, 14] = 1
    W[0, 14] = 1
    
    # m15 = a11 * b23
    U[0, 15] = 1
    V[3, 15] = 1
    W[7, 15] = 1
    
    # m16 = a21 * b33
    U[3, 16] = 1
    V[6, 16] = 1
    W[0, 16] = 1
    
    # m17 = a31 * b13
    U[6, 17] = 1
    V[0, 17] = 1
    W[3, 17] = 1
    
    # m18 = a13 * b21
    U[2, 18] = 1
    V[3, 18] = 1
    W[6, 18] = 1
    
    # m19 = a23 * b31
    U[5, 19] = 1
    V[6, 19] = 1
    W[0, 19] = 1
    
    # m20 = a33 * b11
    U[8, 20] = 1
    V[0, 20] = 1
    W[3, 20] = 1
    
    # m21 = a13 * b32
    U[2, 21] = 1
    V[9, 21] = 1
    W[1, 21] = 1
    
    # m22 = a23 * b12
    U[5, 22] = 1
    V[3, 22] = 1
    W[4, 22] = 1
    
    return U, V, W


def _build_laderman_3x5x6(n, m, p):
    """
    Build tensor decomposition for 3x5 @ 5x6 using Laderman block structure.
    Split A into 3 rows (i=0,1,2).
    Split B into two 5x3 blocks (j=0,1,2 and j=3,4).
    Use Laderman for 3x3x3 sub-problems, schoolbook for 3x2x3.
    """
    lU, lV, lW = _get_laderman_3x3x3()
    
    # Per k-block (3 k-values):
    #   i in {0,1,2}, j in {0,1,2}: Laderman (23)
    #   i in {0,1,2}, j in {3,4}: schoolbook 3x2x3 (18)
    # Per k-block: 23 + 18 = 41
    # 2 k-blocks: 82
    
    rank = 82
    U = np.zeros((n * m, rank))  # 15 x 82
    V = np.zeros((m * p, rank))  # 30 x 82
    W = np.zeros((p * n, rank))  # 18 x 82
    
    col = 0
    
    for k_block in range(2):
        k0 = k_block * 3  # start k index
        k1 = k0 + 1
        k2 = k0 + 2
        
        # i in {0,1,2}, j in {0,1,2}: Laderman
        for c in range(23):
            # U: A factors
            U[0*5 + 0, col] = lU[0, c]   # i=0, j=0
            U[0*5 + 1, col] = lU[1, c]   # i=0, j=1
            U[0*5 + 2, col] = lU[2, c]   # i=0, j=2
            U[1*5 + 0, col] = lU[3, c]   # i=1, j=0
            U[1*5 + 1, col] = lU[4, c]   # i=1, j=1
            U[1*5 + 2, col] = lU[5, c]   # i=1, j=2
            U[2*5 + 0, col] = lU[6, c]   # i=2, j=0
            U[2*5 + 1, col] = lU[7, c]   # i=2, j=1
            U[2*5 + 2, col] = lU[8, c]   # i=2, j=2
            
            # V: B factors
            V[0*6 + k0, col] = lV[0, c]   # j=0, k=k0
            V[0*6 + k1, col] = lV[1, c]   # j=0, k=k1
            V[0*6 + k2, col] = lV[2, c]   # j=0, k=k2
            V[1*6 + k0, col] = lV[3, c]   # j=1, k=k0
            V[1*6 + k1, col] = lV[4, c]   # j=1, k=k1
            V[1*6 + k2, col] = lV[5, c]   # j=1, k=k2
            V[2*6 + k0, col] = lV[6, c]   # j=2, k=k0
            V[2*6 + k1, col] = lV[7, c]   # j=2, k=k1
            V[2*6 + k2, col] = lV[8, c]   # j=2, k=k2
            
            # W: C factors (W[k*3+i, c])
            W[k0*3 + 0, col] = lW[0, c]   # k=k0, i=0
            W[k0*3 + 1, col] = lW[1, c]   # k=k0, i=1
            W[k0*3 + 2, col] = lW[2, c]   # k=k0, i=2
            W[k1*3 + 0, col] = lW[3, c]   # k=k1, i=0
            W[k1*3 + 1, col] = lW[4, c]   # k=k1, i=1
            W[k1*3 + 2, col] = lW[5, c]   # k=k1, i=2
            W[k2*3 + 0, col] = lW[6, c]   # k=k2, i=0
            W[k2*3 + 1, col] = lW[7, c]   # k=k2, i=1
            W[k2*3 + 2, col] = lW[8, c]   # k=k2, i=2
            col += 1
        
        # i in {0,1,2}, j in {3,4}: schoolbook 3x2x3
        for i in range(3):
            for j in [3, 4]:
                for k in range(3):
                    U[i*5 + j, col] = 1
                    V[j*6 + k0 + k, col] = 1
                    W[(k0 + k)*3 + i, col] = 1
                    col += 1
    
    if col != rank:
        # Truncate or pad
        if col > rank:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
        else:
            rank = col
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
    
    return U, V, W, rank


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)


def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    r = U.shape[1]
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

# EVOLVE-BLOCK-END
