import time
import sys

DIMENSIONS = [2, 3, 7]

def _build_tensor():
    """Build the matrix multiplication tensor T[n*m*p] as a flat list."""
    n, m, p = DIMENSIONS
    size = n * m * p
    T = [0.0] * size
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m * p + j * p + k] = 1.0
    return T, n, m, p

def _apply_factors(U, V, W, n, m, p):
    """Apply factor matrices to produce the reconstructed tensor."""
    rank = len(U[0]) if U else 0
    nmp = n * m * p
    result = [0.0] * nmp
    for r in range(rank):
        u_col = [U[i][r] for i in range(n * m)]
        v_col = [V[j][r] for j in range(m * p)]
        w_col = [W[k][r] for k in range(p * n)]
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    if v_val == 0:
                        continue
                    w_val = w_col[k * n + i]
                    result[i * m * p + j * p + k] += u_val * v_val * w_val
    return result

def _verify(U, V, W, n, m, p, T):
    """Verify that the decomposition is exact."""
    result = _apply_factors(U, V, W, n, m, p)
    for i in range(len(T)):
        if abs(result[i] - T[i]) > 1e-9:
            return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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
    return U, V, W, rank

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7.
    
    The 7 rank-1 terms correspond to:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B22)
    M4 = A22(B12 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # n=2, m=2, p=2
    # U shape: (2*2, 7) = (4, 7)
    # V shape: (2*2, 7) = (4, 7)
    # W shape: (2*2, 7) = (4, 7)
    
    # Indexing: U[i*m + j], V[j*p + k], W[k*n + i]
    # i in {0,1}, j in {0,1}, k in {0,1}
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: A00 += 1, A11 += 1
    # V: B00 += 1, B11 += 1
    # W: C00 += 1, C11 += 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10 + A11)B00
    # U: A10 += 1, A11 += 1
    # V: B00 += 1
    # W: C10 += 1, C11 -= 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10 - B11)
    # U: A00 += 1
    # V: B10 += 1, B11 -= 1
    # W: C01 += 1, C11 += 1
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01 - B00)
    # U: A11 += 1
    # V: B01 += 1, B00 -= 1
    # W: C00 += 1, C10 += 1
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00 + A01)B11
    # U: A00 += 1, A01 += 1
    # V: B11 += 1
    # W: C00 -= 1, C01 += 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    # U: A10 += 1, A00 -= 1
    # V: B00 += 1, B01 += 1
    # W: C11 += 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    # U: A01 += 1, A11 -= 1
    # V: B10 += 1, B11 += 1
    # W: C00 += 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W, 7

def _try_reduce_rank(U, V, W, n, m, p, T, max_time=10.0):
    """
    Try to reduce rank by using block decomposition with Strassen for 2x2x2 blocks.
    
    For 2x3x7:
    - Split the middle dimension (m=3) into blocks of size 1 and 2.
    - For m=1 blocks, use schoolbook.
    - For m=2 blocks, further split p=7 into pairs of 2 and a remainder of 1.
      - Use Strassen for 2x2x2 blocks.
      - Use schoolbook for 2x2x1 blocks.
    
    Block structure:
    - j=0: m_block=1, p=7. Schoolbook 2x1x7. Rank = 2*1*7 = 14.
    - j=1,2: m_block=2, p=7.
      - Split p into: (0,1), (2,3), (4,5), (6)
      - (0,1): Strassen 2x2x2. Rank = 7.
      - (2,3): Strassen 2x2x2. Rank = 7.
      - (4,5): Strassen 2x2x2. Rank = 7.
      - (6): Schoolbook 2x2x1. Rank = 2*2*1 = 4.
      - Total for j=1,2: 7+7+7+4 = 25.
    
    Total rank = 14 + 25 = 39.
    """
    start_time = time.time()
    
    # Initialize global factor matrices
    n, m, p = DIMENSIONS
    total_rank = 14 + 25  # 39
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j=0, m_block=1, p=7
    # Schoolbook 2x1x7
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2: j=1,2, m_block=2, p=7
    # Split p into pairs: (0,1), (2,3), (4,5), (6)
    
    # Sub-block 2a: j=1,2; k=0,1. Strassen 2x2x2.
    sU, sV, sW, srank = _strassen_2x2x2()
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + (1 + j)][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[(1 + j) * p + k][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[k * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 2b: j=1,2; k=2,3. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + (1 + j)][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[(1 + j) * p + (2 + k)][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(2 + k) * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 2c: j=1,2; k=4,5. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + (1 + j)][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[(1 + j) * p + (4 + k)][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(4 + k) * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 2d: j=1,2; k=6. Schoolbook 2x2x1.
    for i in range(2):
        for j in range(2):
            U[i * m + (1 + j)][col] = 1
            V[(1 + j) * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1
    
    # Verify
    if not _verify(U, V, W, n, m, p, T):
        # Fallback to schoolbook
        U, V, W, rank = _schoolbook(n, m, p)
        return U, V, W, rank
    
    return U, V, W, total_rank

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    T, n, m, p = _build_tensor()
    
    # Start with schoolbook decomposition
    U, V, W, rank = _schoolbook(n, m, p)
    
    # Try to improve with block decomposition
    if time.time() < deadline:
        U, V, W, rank = _try_reduce_rank(U, V, W, n, m, p, T, max_time=min(deadline - time.time(), 50.0))
    
    return {"U": U, "V": V, "W": W}
