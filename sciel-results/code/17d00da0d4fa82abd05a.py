"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]
import time

# EVOLVE-BLOCK-START
def _get_strassen_2x2x2():
    """
    Return (U, V, W) as 3D arrays of shape (2,2,7) for 2x2x2 matrix multiplication.
    U[i,j,t] = coefficient of A[i,j] in Strassen term t
    V[j,k,t] = coefficient of B[j,k] in Strassen term t
    W[k,i,t] = coefficient of C[k,i] in Strassen term t
    such that sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] = 1 for all i,j,k.
    """
    U = [[0]*7 for _ in range(4)]  # indexed by i*2+j
    V = [[0]*7 for _ in range(4)]  # indexed by j*2+k
    W = [[0]*7 for _ in range(4)]  # indexed by k*2+i

    # Strassen's 7 products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)

    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P3 + P6 - P2

    # Term 0 (P1): (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1  # A[0,0], A[1,1]
    V[0][0] = 1; V[3][0] = 1  # B[0,0], B[1,1]
    W[0][0] = 1; W[2][0] = 1  # C[0,0], C[1,0]
    W[1][0] = -1; W[3][0] = -1  # C[0,1], C[1,1]

    # Term 1 (P2): (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1  # A[1,0], A[1,1]
    V[0][1] = 1  # B[0,0]
    W[2][1] = 1; W[3][1] = -1  # C[1,0], C[1,1]

    # Term 2 (P3): A00(B10-B11)
    U[0][2] = 1  # A[0,0]
    V[2][2] = 1; V[3][2] = -1  # B[1,0], B[1,1]
    W[1][2] = 1; W[3][2] = 1  # C[0,1], C[1,1]

    # Term 3 (P4): A11(B01-B00)
    U[3][3] = 1  # A[1,1]
    V[1][3] = 1; V[0][3] = -1  # B[0,1], B[0,0]
    W[0][3] = 1; W[2][3] = 1  # C[0,0], C[1,0]

    # Term 4 (P5): (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1  # A[0,0], A[0,1]
    V[3][4] = 1  # B[1,1]
    W[1][4] = 1; W[3][4] = 1  # C[0,1], C[1,1]

    # Term 5 (P6): (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1  # A[1,0], A[0,0]
    V[0][5] = 1; V[1][5] = 1  # B[0,0], B[0,1]
    W[3][5] = 1  # C[1,1]

    # Term 6 (P7): (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1  # A[0,1], A[1,1]
    V[2][6] = 1; V[3][6] = 1  # B[1,0], B[1,1]
    W[0][6] = 1  # C[0,0]

    return U, V, W

def _verify_strassen_block(U, V, W):
    """Verify that the Strassen block decomposition is correct."""
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0
                for t in range(7):
                    val += U[i*2+j][t] * V[j*2+k][t] * W[k*2+i][t]
                if val != 1:
                    return False
    return True

def _build_strassen_49():
    """Build the full rank-49 Strassen-based decomposition for 2x2 x 2x14."""
    n, m, p = 2, 2, 14
    num_blocks = p // m  # 7 blocks of 2 columns each
    
    U_blk, V_blk, W_blk = _get_strassen_2x2x2()
    
    if not _verify_strassen_block(U_blk, V_blk, W_blk):
        return None
    
    rank = 7 * num_blocks  # 49
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49
    
    for b in range(num_blocks):
        for t in range(7):
            col = b * 7 + t
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = U_blk[i * m + j][t]
            for j in range(2):
                for k in range(2):
                    V[j * p + b * m + k][col] = V_blk[j * m + k][t]
            for k in range(2):
                for i in range(2):
                    W[(b * m + k) * n + i][col] = W_blk[k * n + i][t]
    
    # Full verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if val != 1:
                    return None
    
    return {"U": U, "V": V, "W": W}

def _build_schoolbook():
    """Build the schoolbook rank-56 decomposition as fallback."""
    n, m, p = 2, 2, 14
    rank = n * m * p  # 56
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
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Try Strassen-based decomposition (rank 49)
    try:
        result = _build_strassen_49()
        if result is not None:
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 56)
    return _build_schoolbook()
# EVOLVE-BLOCK-END
