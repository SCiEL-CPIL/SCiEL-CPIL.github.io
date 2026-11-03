import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def _build_strassen_2x3x8():
    """
    Build a rank-44 decomposition for 2x3 * 3x8 matrix multiplication.
    
    Structure:
    - j=0: 16 schoolbook terms (2 i-values * 8 k-values)
    - j=1,2: 4 blocks of 2x2x2, each using Strassen (7 terms) = 28 terms
    Total: 16 + 28 = 44
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    col = 0
    
    # Part 1: j=0, schoolbook for 2x8
    # For each i in {0,1} and k in {0,...,7}:
    #   U[i*3+0][col] = 1  (A[i,0])
    #   V[0*8+k][col] = 1  (B[0,k])
    #   W[k*2+i][col] = 1  (C[i,k])
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: j=1,2 in 4 blocks of 2x2x2 using Strassen
    # For each block of 2 k-values (k0, k1):
    #   We're computing contributions from A[:,1] and A[:,2] times B[1:3,k0:k1+1]
    #   This is a 2x2 * 2x2 problem (after reindexing)
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen for 2x2x2:
        # The 2x2 A-block: rows i=0,1, cols j=1,2
        # The 2x2 B-block: rows j=1,2, cols k=k0,k1
        # Output 2x2 C-block: rows i=0,1, cols k=k0,k1
        
        # m1 = (a00+a11)(b00+b11)
        # a00 = A[0,1], a11 = A[1,2], b00 = B[1,k0], b11 = B[2,k1]
        # Contribution to C[0,k0], C[0,k1], C[1,k0], C[1,k1]
        U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1; V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        col += 1
        
        # m2 = (a10+a11)b00
        # a10 = A[1,1], a11 = A[1,2], b00 = B[1,k0]
        # Contribution to C[1,k0] with +, C[1,k1] with -
        U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1; W[k1 * n + 1][col] = -1
        col += 1
        
        # m3 = a00(b01-b11)
        # a00 = A[0,1], b01 = B[1,k1], b11 = B[2,k1]
        # Contribution to C[0,k1] with +, C[1,k1] with +
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1; V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        col += 1
        
        # m4 = a11(b10-b00)
        # a11 = A[1,2], b10 = B[2,k0], b00 = B[1,k0]
        # Contribution to C[0,k0] with +, C[1,k0] with +
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1; V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1; W[k0 * n + 1][col] = 1
        col += 1
        
        # m5 = (a00+a01)b11
        # a00 = A[0,1], a01 = A[0,2], b11 = B[2,k1]
        # Contribution to C[0,k0] with -, C[0,k1] with +
        U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1; W[k1 * n + 0][col] = 1
        col += 1
        
        # m6 = (a10-a00)(b00+b01)
        # a10 = A[1,1], a00 = A[0,1], b00 = B[1,k0], b01 = B[1,k1]
        # Contribution to C[1,k1] with +
        U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1; V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # m7 = (a01-a11)(b10+b11)
        # a01 = A[0,2], a11 = A[1,2], b10 = B[2,k0], b11 = B[2,k1]
        # Contribution to C[0,k0] with +
        U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1; V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
    
    return U, V, W, rank

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_true[a_idx, b_idx, c_idx] = 1.0
    
    return np.allclose(T_decomp, T_true, atol=1e-10)

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank

def construct():
    n, m, p = 2, 3, 8
    
    # Use the rank-44 Strassen-based construction
    U, V, W, rank = _build_strassen_2x3x8()
    
    # Verify with fallback to schoolbook if verification fails
    if time.time() - _START_TIME < TIME_LIMIT:
        if not _verify_decomposition(U, V, W, n, m, p):
            U, V, W, rank = _build_schoolbook(n, m, p)
        
    return {"U": U, "V": V, "W": W}
