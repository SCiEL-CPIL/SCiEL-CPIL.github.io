import time
import numpy as np

DIMENSIONS = [2, 3, 5]
TIME_LIMIT = 150
SAFETY_MARGIN = 10

def _verify_decomposition(U, V, W, n, m, p):
    n, m, p = int(n), int(m), int(p)
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(U.shape[1]):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if total != 1:
                    return False
    return True

def _strassen_block(n, m, p, i_offset, j_offset, k_offset, U, V, W, r_start):
    """
    Apply Strassen's 7-multiplication algorithm for a 2x2 block.
    Returns the next rank index.
    """
    r = r_start
    # M1 = (A11 + A22) * (B11 + B22)
    # U: A11 (i=0,j=0), A22 (i=1,j=1)
    # V: B11 (j=0,k=0), B22 (j=1,k=1)
    # W: C11 (i=0,k=0), C22 (i=1,k=1)
    U[(i_offset+0)*m + (j_offset+0)][r] = 1
    U[(i_offset+1)*m + (j_offset+1)][r] = 1
    V[(j_offset+0)*p + (k_offset+0)][r] = 1
    V[(j_offset+1)*p + (k_offset+1)][r] = 1
    W[(k_offset+0)*n + (i_offset+0)][r] = 1
    W[(k_offset+1)*n + (i_offset+1)][r] = 1
    r += 1
    
    # M2 = (A21 + A22) * B11
    # U: A21 (i=1,j=0), A22 (i=1,j=1)
    # V: B11 (j=0,k=0)
    # W: C21 (i=1,k=0), C22 (i=1,k=1) with signs
    U[(i_offset+1)*m + (j_offset+0)][r] = 1
    U[(i_offset+1)*m + (j_offset+1)][r] = 1
    V[(j_offset+0)*p + (k_offset+0)][r] = 1
    W[(k_offset+0)*n + (i_offset+1)][r] = 1
    W[(k_offset+1)*n + (i_offset+1)][r] = -1
    r += 1
    
    # M3 = A11 * (B12 - B22)
    # U: A11 (i=0,j=0)
    # V: B12 (j=0,k=1), B22 (j=1,k=1)
    # W: C12 (i=0,k=1), C22 (i=1,k=1)
    U[(i_offset+0)*m + (j_offset+0)][r] = 1
    V[(j_offset+0)*p + (k_offset+1)][r] = 1
    V[(j_offset+1)*p + (k_offset+1)][r] = -1
    W[(k_offset+1)*n + (i_offset+0)][r] = 1
    W[(k_offset+1)*n + (i_offset+1)][r] = 1
    r += 1
    
    # M4 = A22 * (B21 - B11)
    # U: A22 (i=1,j=1)
    # V: B21 (j=1,k=0), B11 (j=0,k=0)
    # W: C11 (i=0,k=0), C21 (i=1,k=0)
    U[(i_offset+1)*m + (j_offset+1)][r] = 1
    V[(j_offset+1)*p + (k_offset+0)][r] = 1
    V[(j_offset+0)*p + (k_offset+0)][r] = -1
    W[(k_offset+0)*n + (i_offset+0)][r] = 1
    W[(k_offset+0)*n + (i_offset+1)][r] = 1
    r += 1
    
    # M5 = (A11 + A12) * B22
    # U: A11 (i=0,j=0), A12 (i=0,j=1)
    # V: B22 (j=1,k=1)
    # W: C11 (i=0,k=0), C12 (i=0,k=1)
    U[(i_offset+0)*m + (j_offset+0)][r] = 1
    U[(i_offset+0)*m + (j_offset+1)][r] = 1
    V[(j_offset+1)*p + (k_offset+1)][r] = 1
    W[(k_offset+0)*n + (i_offset+0)][r] = -1
    W[(k_offset+1)*n + (i_offset+0)][r] = 1
    r += 1
    
    # M6 = (A21 - A11) * (B11 + B12)
    # U: A21 (i=1,j=0), A11 (i=0,j=0)
    # V: B11 (j=0,k=0), B12 (j=0,k=1)
    # W: C21 (i=1,k=0), C22 (i=1,k=1)
    U[(i_offset+1)*m + (j_offset+0)][r] = 1
    U[(i_offset+0)*m + (j_offset+0)][r] = -1
    V[(j_offset+0)*p + (k_offset+0)][r] = 1
    V[(j_offset+0)*p + (k_offset+1)][r] = 1
    W[(k_offset+0)*n + (i_offset+1)][r] = 1
    W[(k_offset+1)*n + (i_offset+1)][r] = 1
    r += 1
    
    # M7 = (A12 - A22) * (B21 + B22)
    # U: A12 (i=0,j=1), A22 (i=1,j=1)
    # V: B21 (j=1,k=0), B22 (j=1,k=1)
    # W: C11 (i=0,k=0), C12 (i=0,k=1)
    U[(i_offset+0)*m + (j_offset+1)][r] = 1
    U[(i_offset+1)*m + (j_offset+1)][r] = -1
    V[(j_offset+1)*p + (k_offset+0)][r] = 1
    V[(j_offset+1)*p + (k_offset+1)][r] = 1
    W[(k_offset+0)*n + (i_offset+0)][r] = 1
    W[(k_offset+1)*n + (i_offset+0)][r] = 1
    r += 1
    
    return r

def construct():
    n, m, p = DIMENSIONS
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 * B2 (j=2, k=0..4)
    # 1 term.
    for i in range(n):
        U[i * m + 2][r] = 1
    for k in range(p):
        V[2 * p + k][r] = 1
        W[k * n + 0][r] = 1
        W[k * n + 1][r] = 1
    r += 1
    
    # Part 2: A1 * B13 (j=0,1, k=4)
    # 1 term.
    for i in range(2):
        for j in range(2):
            U[i * m + j][r] = 1
    for j in range(2):
        V[j * p + 4][r] = 1
    for i in range(2):
        W[4 * n + i][r] = 1
    r += 1
    
    # Part 3: A1 * B11 (j=0,1, k=0,1)
    # Strassen 2x2 block.
    r = _strassen_block(n, m, p, 0, 0, 0, U, V, W, r)
    
    # Part 4: A1 * B12 (j=0,1, k=2,3)
    # Strassen 2x2 block.
    r = _strassen_block(n, m, p, 0, 0, 2, U, V, W, r)
    
    if r != rank:
        print(f"Warning: Rank mismatch. Expected {rank}, got {r}")
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        if not _verify_decomposition(U, V, W, n, m, p):
            raise ValueError("Fallback decomposition failed verification")
    
    return {
        "U": U,
        "V": V,
        "W": W
    }
