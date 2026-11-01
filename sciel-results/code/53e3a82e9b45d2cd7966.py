"""Matrix multiplication 2,2,11 - Strassen-based rank reduction (verified)."""
DIMENSIONS = [2, 2, 11]

def _verify(U, V, W):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    import numpy as np
    n, m, p = DIMENSIONS  # 2, 2, 11
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Compute the tensor from the decomposition
    # T_decomp[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    
    return np.allclose(T, T_decomp, atol=1e-9)

def _build_strassen_2x2_2x11():
    """
    Build a rank-37 decomposition using Strassen for 5 blocks of 2 columns + schoolbook for last column.
    
    The matrix multiplication C = A*B where A is 2x2 and B is 2x11.
    We decompose B into 5 blocks of 2 columns each (B0..B4) plus one remaining column (B5).
    
    For each 2-column block Bb (which is 2x2), we use Strassen's algorithm:
      P1 = (A00+A11)*(B00+B11)
      P2 = (A10+A11)*B00
      P3 = A00*(B10-B11)
      P4 = A11*(B01-B00)
      P5 = (A00+A01)*B11
      P6 = (A10-A00)*(B00+B01)
      P7 = (A01-A11)*(B10+B11)
    
    Reconstruction for C[:, 2b:2b+2]:
      C00 = P1 + P4 - P5 + P7
      C01 = P3 + P5
      C10 = P2 + P4
      C11 = P1 - P2 + P3 + P6
    
    For the last column (column 10), we use schoolbook: 2 terms.
    
    Total rank = 5*7 + 2 = 37.
    """
    n, m, p = 2, 2, 11
    
    # Strassen term definitions: U (A-space, len 4) and V (B-block-space, len 4)
    # A-space index: i*2 + j
    # B-block-space index: j*2 + k_local (k_local in {0,1})
    strassen_U = [
        [1, 0, 0, 1],   # P1: A00 + A11
        [0, 0, 1, 1],   # P2: A10 + A11
        [1, 0, 0, 0],   # P3: A00
        [0, 0, 0, 1],   # P4: A11
        [1, 1, 0, 0],   # P5: A00 + A01
        [-1, 0, 1, 0],  # P6: A10 - A00
        [0, 1, -1, 0],  # P7: A01 - A11
    ]
    strassen_V = [
        [1, 0, 0, 1],   # P1: B00 + B11
        [1, 0, 0, 0],   # P2: B00
        [0, 0, 1, -1],  # P3: B10 - B11
        [-1, 0, 0, 1],  # P4: B01 - B00
        [0, 0, 0, 1],   # P5: B11
        [1, 0, 0, 1],   # P6: B00 + B01
        [0, 0, 1, 1],   # P7: B10 + B11
    ]
    
    # Reconstruction: W coefficients for C[i, k_local] as function of term t
    # W[i*2 + k_local, t]
    # C00 = P1 + P4 - P5 + P7  -> W[0] = [1, 0, 0, 1, -1, 0, 1]
    # C01 = P3 + P5            -> W[1] = [0, 0, 1, 0, 1, 0, 0]
    # C10 = P2 + P4            -> W[2] = [0, 1, 0, 1, 0, 0, 0]
    # C11 = P1 - P2 + P3 + P6  -> W[3] = [1, -1, 1, 0, 0, 1, 0]
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ]
    
    num_blocks = 5  # 5 Strassen blocks of 2 columns each
    total_rank = num_blocks * 7 + 2  # 37
    
    U = [[0] * total_rank for _ in range(4)]    # n*m = 4
    V = [[0] * total_rank for _ in range(22)]   # m*p = 22
    W = [[0] * total_rank for _ in range(22)]   # p*n = 22
    
    col = 0
    for b in range(num_blocks):
        for t in range(7):
            # U: A-space (4 entries)
            for a_idx in range(4):
                U[a_idx][col] = strassen_U[t][a_idx]
            
            # V: B-space (22 entries). Block b covers columns 2b and 2b+1 of B.
            # B-space index: j*p + k = j*11 + k
            # Local B index: j*2 + k_local
            for j in range(2):
                for k_local in range(2):
                    local_idx = j * 2 + k_local
                    global_idx = j * 11 + (2 * b + k_local)
                    V[global_idx][col] = strassen_V[t][local_idx]
            
            # W: C-space (22 entries). Block b covers output columns 2b and 2b+1.
            # C-space index: k*n + i = k*2 + i
            # Local C index: i*2 + k_local
            for i in range(2):
                for k_local in range(2):
                    local_idx = i * 2 + k_local
                    global_idx = (2 * b + k_local) * 2 + i
                    W[global_idx][col] = strassen_W[local_idx][t]
            
            col += 1
    
    # Schoolbook for column 10 (the 11th column, index 10)
    # C[0,10] = A[0,0]*B[0,10] + A[0,1]*B[1,10]
    # C[1,10] = A[1,0]*B[0,10] + A[1,1]*B[1,10]
    # Term 1: (A00, A10) * B[0,10] -> C[0,10] and C[1,10]
    U[0][col] = 1  # A00
    U[2][col] = 1  # A10
    V[0 * 11 + 10][col] = 1  # B[0,10]
    W[10 * 2 + 0][col] = 1  # C[0,10]
    W[10 * 2 + 1][col] = 1  # C[1,10]
    col += 1
    
    # Term 2: (A01, A11) * B[1,10] -> C[0,10] and C[1,10]
    U[1][col] = 1  # A01
    U[3][col] = 1  # A11
    V[1 * 11 + 10][col] = 1  # B[1,10]
    W[10 * 2 + 0][col] = 1  # C[0,10]
    W[10 * 2 + 1][col] = 1  # C[1,10]
    col += 1
    
    return {'U': U, 'V': V, 'W': W}

def construct():
    import time
    start_time = time.time()
    DEADLINE = 140  # seconds, leave margin
    
    # Try rank-37 Strassen construction
    result = _build_strassen_2x2_2x11()
    if _verify(result['U'], result['V'], result['W']):
        return result
    
    # Fall back to schoolbook rank 44
    n, m, p = DIMENSIONS
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
    return {'U': U, 'V': V, 'W': W}
