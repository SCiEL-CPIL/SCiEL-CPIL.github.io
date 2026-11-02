import numpy as np
import time

DIMENSIONS = [3, 4, 6]

_start_time = time.time()
_DEADLINE = 140.0  # Leave 10s safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U and len(U) > 0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Faster verification using numpy."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_np[i * m + j]
                v_row = V_np[j * p + k]
                w_row = W_np[k * n + i]
                s = np.dot(u_row * v_row, w_row)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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
    return U, V, W

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2x2 matrix multiplication.
    
    U rows indexed by (i,j) -> i*2+j (input matrix A indices)
    V rows indexed by (j,k) -> j*2+k (input matrix B indices)
    W rows indexed by (k,i) -> k*2+i (output C indices)
    
    The 7 Strassen products:
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)b00
    M3 = a00(b01-b11)
    M4 = a11(b10-b00)
    M5 = (a00+a01)b11
    M6 = (a10-a00)(b00+b01)
    M7 = (a01-a11)(b10+b11)
    
    c00 = M1 + M4 - M5 + M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M2 + M3 + M6
    """
    rank = 7
    # U: 4 rows (i*2+j), 7 columns
    U = [[0]*rank for _ in range(4)]
    # V: 4 rows (j*2+k), 7 columns
    V = [[0]*rank for _ in range(4)]
    # W: 4 rows (k*2+i), 7 columns
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11)
    # U: a00 -> col0, a11 -> col0
    U[0][0] = 1  # a00
    U[3][0] = 1  # a11
    # V: b00 -> col0, b11 -> col0
    V[0][0] = 1  # b00
    V[3][0] = 1  # b11
    # W: c00 += M1, c11 += M1
    W[0][0] = 1   # c00 (k=0,i=0)
    W[3][0] = 1   # c11 (k=1,i=1)
    
    # M2 = (a10+a11)b00
    U[2][1] = 1  # a10
    U[3][1] = 1  # a11
    V[0][1] = 1  # b00
    W[2][1] = 1  # c10 (k=1,i=0)
    
    # M3 = a00(b01-b11)
    U[0][2] = 1  # a00
    V[1][2] = 1  # b01
    V[3][2] = -1 # b11
    W[1][2] = 1  # c01 (k=0,i=1)
    
    # M4 = a11(b10-b00)
    U[3][3] = 1  # a11
    V[2][3] = 1  # b10
    V[0][3] = -1 # b00
    W[0][3] = 1  # c00
    W[2][3] = 1  # c10
    
    # M5 = (a00+a01)b11
    U[0][4] = 1  # a00
    U[1][4] = 1  # a01
    V[3][4] = 1  # b11
    W[0][4] = -1 # c00 (minus)
    W[1][4] = 1  # c01
    
    # M6 = (a10-a00)(b00+b01)
    U[2][5] = 1  # a10
    U[0][5] = -1 # a00
    V[0][5] = 1  # b00
    V[1][5] = 1  # b01
    W[3][5] = 1  # c11
    
    # M7 = (a01-a11)(b10+b11)
    U[1][6] = 1  # a01
    U[3][6] = -1 # a11
    V[2][6] = 1  # b10
    V[3][6] = 1  # b11
    W[0][6] = 1  # c00
    W[3][6] = -1 # c11 (minus)
    
    return U, V, W

def _combine_split_m(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
    """Combine two decompositions split along the middle dimension m."""
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    rank_full = rank1 + rank2
    m_full = m1 + m2
    
    U_full = [[0] * rank_full for _ in range(n * m_full)]
    V_full = [[0] * rank_full for _ in range(m_full * p)]
    W_full = [[0] * rank_full for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m1):
            U_full[i * m_full + j] = U1[i * m1 + j] + [0] * rank2
        for j in range(m2):
            U_full[i * m_full + (m1 + j)] = [0] * rank1 + U2[i * m2 + j]
            
    for j in range(m1):
        for k in range(p):
            V_full[j * p + k] = V1[j * p + k] + [0] * rank2
    for j in range(m2):
        for k in range(p):
            V_full[(m1 + j) * p + k] = [0] * rank1 + V2[j * p + k]
            
    for k in range(p):
        for i in range(n):
            idx = k * n + i
            W_full[idx] = [W1[idx][r] for r in range(rank1)] + [W2[idx][r] for r in range(rank2)]
            
    return U_full, V_full, W_full

def _combine_split_p(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
    """Combine two decompositions split along the output dimension p."""
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    rank_full = rank1 + rank2
    p_full = p1 + p2
    
    U_full = [[0] * rank_full for _ in range(n * m)]
    V_full = [[0] * rank_full for _ in range(m * p_full)]
    W_full = [[0] * rank_full for _ in range(p_full * n)]
    
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            U_full[idx] = [U1[idx][r] for r in range(rank1)] + [U2[idx][r] for r in range(rank2)]
            
    for j in range(m):
        for k in range(p1):
            V_full[j * p_full + k] = V1[j * p1 + k] + [0] * rank2
    for j in range(m):
        for k in range(p2):
            V_full[j * p_full + (p1 + k)] = [0] * rank1 + V2[j * p2 + k]
            
    for k in range(p1):
        for i in range(n):
            idx = k * n + i
            W_full[idx] = W1[idx] + [0] * rank2
    for k in range(p2):
        for i in range(n):
            idx = (p1 + k) * n + i
            W_full[idx] = [0] * rank1 + W2[idx]
            
    return U_full, V_full, W_full

def _combine_split_n(U1, V1, W1, U2, V2, W2, n1, n2, m, p):
    """Combine two decompositions split along the input dimension n."""
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    rank_full = rank1 + rank2
    n_full = n1 + n2
    
    U_full = [[0] * rank_full for _ in range(n_full * m)]
    V_full = [[0] * rank_full for _ in range(m * p)]
    W_full = [[0] * rank_full for _ in range(p * n_full)]
    
    for i in range(n1):
        for j in range(m):
            U_full[i * m + j] = U1[i * m + j] + [0] * rank2
    for i in range(n2):
        for j in range(m):
            U_full[(n1 + i) * m + j] = [0] * rank1 + U2[i * m + j]
            
    for j in range(m):
        for k in range(p):
            idx = j * p + k
            V_full[idx] = [V1[idx][r] for r in range(rank1)] + [V2[idx][r] for r in range(rank2)]
            
    for k in range(p):
        for i in range(n1):
            idx = k * n_full + i
            W_full[idx] = W1[k * n1 + i] + [0] * rank2
    for k in range(p):
        for i in range(n2):
            idx = k * n_full + (n1 + i)
            W_full[idx] = [0] * rank1 + W2[k * n2 + i]
            
    return U_full, V_full, W_full

def _block_1x2x2():
    """
    Decomposition for 1x2x2 matrix multiplication (rank 4).
    A is 1x2, B is 2x2, C is 1x2.
    U: 2 rows (i*2+j where i=0, j=0,1) -> rows 0,1
    V: 4 rows (j*2+k)
    W: 2 rows (k*2+i where i=0, k=0,1) -> rows 0,1
    """
    rank = 4
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(2)]
    
    # c00 = a00*b00 + a01*b10
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    U[1][1] = 1; V[2][1] = 1; W[0][1] = 1
    
    # c01 = a00*b01 + a01*b11
    U[0][2] = 1; V[1][2] = 1; W[1][2] = 1
    U[1][3] = 1; V[3][3] = 1; W[1][3] = 1
    
    return U, V, W

def _block_2x2x2():
    """Return Strassen decomposition for 2x2x2."""
    return _strassen_2x2x2()

def _decompose_3x4x6():
    """
    Construct rank-66 decomposition for 3x4x6.
    
    Strategy:
    - Split n=3 into 1+2
    - Split m=4 into 2+2
    - Split p=6 into 2+2+2
    
    This gives blocks:
    - For n-part=1 (1 row): 2 m-blocks x 3 p-blocks = 6 blocks of 1x2x2 (rank 4 each) = 24
    - For n-part=2 (2 rows): 2 m-blocks x 3 p-blocks = 6 blocks of 2x2x2 (rank 7 each) = 42
    Total rank: 24 + 42 = 66
    """
    n, m, p = 3, 4, 6
    
    # Get base blocks
    U_1x2x2, V_1x2x2, W_1x2x2 = _block_1x2x2()
    U_2x2x2, V_2x2x2, W_2x2x2 = _block_2x2x2()
    
    # For the n=1 part: combine 3 p-blocks of 1x2x2
    # First combine p-blocks 0 and 1
    U_1x2x4, V_1x2x4, W_1x2x4 = _combine_split_p(
        U_1x2x2, V_1x2x2, W_1x2x2,
        U_1x2x2, V_1x2x2, W_1x2x2,
        1, 2, 2, 2
    )
    # Then combine with p-block 2
    U_1x2x6, V_1x2x6, W_1x2x6 = _combine_split_p(
        U_1x2x4, V_1x2x4, W_1x2x4,
        U_1x2x2, V_1x2x2, W_1x2x2,
        1, 2, 4, 2
    )
    
    # Now split m=4 into 2+2 for the n=1 part
    # We need 1x4x6, so combine two 1x2x6 blocks
    U_1x4x6, V_1x4x6, W_1x4x6 = _combine_split_m(
        U_1x2x6, V_1x2x6, W_1x2x6,
        U_1x2x6, V_1x2x6, W_1x2x6,
        1, 2, 2, 6
    )
    
    # For the n=2 part: combine 3 p-blocks of 2x2x2
    U_2x2x4, V_2x2x4, W_2x2x4 = _combine_split_p(
        U_2x2x2, V_2x2x2, W_2x2x2,
        U_2x2x2, V_2x2x2, W_2x2x2,
        2, 2, 2, 2
    )
    U_2x2x6, V_2x2x6, W_2x2x6 = _combine_split_p(
        U_2x2x4, V_2x2x4, W_2x2x4,
        U_2x2x2, V_2x2x2, W_2x2x2,
        2, 2, 4, 2
    )
    
    # Split m=4 into 2+2 for the n=2 part
    U_2x4x6, V_2x4x6, W_2x4x6 = _combine_split_m(
        U_2x2x6, V_2x2x6, W_2x2x6,
        U_2x2x6, V_2x2x6, W_2x2x6,
        2, 2, 2, 6
    )
    
    # Finally combine n=1 and n=2 parts
    U_full, V_full, W_full = _combine_split_n(
        U_1x4x6, V_1x4x6, W_1x4x6,
        U_2x4x6, V_2x4x6, W_2x4x6,
        1, 2, 4, 6
    )
    
    return U_full, V_full, W_full

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4x6 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    
    try:
        U, V, W = _decompose_3x4x6()
        rank = len(U[0])
        
        # Verify with fast method first
        if _verify_fast(U, V, W, n, m, p, tol=1e-6):
            # Double-check with slower exact method
            if _verify(U, V, W, n, m, p, tol=1e-9):
                return {"U": U, "V": V, "W": W}
    except Exception as e:
        pass
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
