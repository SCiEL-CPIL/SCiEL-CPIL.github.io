"""Strassen-based decomposition for 2x2x7 matrix multiplication."""
import time
import itertools

DIMENSIONS = [2, 2, 7]

def _verify_decomposition(U, V, W, n=2, m=2, p=7):
    """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _strassen_2x2x7():
    """
    Build rank-25 decomposition using Strassen on three 2x2 blocks and schoolbook for the last column.
    
    The 2x2x7 multiplication: A (2x2) * B (2x7) = C (2x7)
    We group B's columns into three pairs (0,1), (2,3), (4,5) and one singleton (6).
    
    For each pair of columns (k, k+1), we apply Strassen to the 2x2x2 sub-problem.
    For the last column (k=6), we use schoolbook (rank 4).
    
    Total rank = 3 * 7 + 4 = 25.
    """
    n, m, p = 2, 2, 7
    rank = 25
    U = [[0] * rank for _ in range(n * m)]  # 4 x 25
    V = [[0] * rank for _ in range(m * p)]  # 14 x 25
    W = [[0] * rank for _ in range(p * n)]  # 14 x 25
    
    # Strassen 2x2x2 formulas:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U vectors (A-side, 4 entries: A00, A01, A10, A11):
    # M1: (1, 0, 0, 1)
    # M2: (0, 0, 1, 1)
    # M3: (1, 0, 0, 0)
    # M4: (0, 0, 0, 1)
    # M5: (1, 1, 0, 0)
    # M6: (-1, 0, 1, 0)
    # M7: (0, 1, 0, -1)
    
    U_strassen = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    
    # V vectors (B-side, for a 2x2 block with columns k, k+1):
    # B indices: B0k, B0(k+1), B1k, B1(k+1)
    # M1: (1, 1, 1, 1)       -> B0k + B0(k+1) for row 0, B1k + B1(k+1) for row 1... 
    # Wait, V is indexed by (j, k) where j is row of B, k is column.
    # For a 2x2 block at columns (k, k+1), the B entries involved are:
    #   B[0, k], B[0, k+1], B[1, k], B[1, k+1]
    # V vectors (4 entries: B0k, B0(k+1), B1k, B1(k+1)):
    # M1: (B0k + B1k) for "B00+B11" pattern... 
    # Let me re-derive carefully.
    # 
    # In the 2x2x2 case, B is 2x2: B[0][0], B[0][1], B[1][0], B[1][1]
    # M1 = (A00+A11)*(B00+B11): V vector = (1, 0, 1, 0) -- no wait.
    # B00 = B[0][0], B11 = B[1][1]
    # So B00+B11 means B[0][0] + B[1][1], so V = (1, 0, 0, 1)
    # M2 = (A10+A11)*B00: V = (1, 0, 0, 0)
    # M3 = A00*(B01-B11): V = (0, 1, 0, -1)
    # M4 = A11*(B10-B00): V = (-1, 0, 1, 0)
    # M5 = (A00+A01)*B11: V = (0, 0, 0, 1)
    # M6 = (A10-A00)*(B00+B01): V = (1, 1, 0, 0)
    # M7 = (A01-A11)*(B10+B11): V = (0, 0, 1, 1)
    
    V_strassen = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01 - B11
        [-1, 0, 1, 0],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1],   # M7: B10 + B11
    ]
    
    # W vectors (C-side, for a 2x2 block with columns k, k+1):
    # C indices: C0k, C0(k+1), C1k, C1(k+1)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # W vector for each M (4 entries: C0k, C0(k+1), C1k, C1(k+1)):
    # M1: (1, 0, 0, 1)   -> contributes to C00 and C11
    # M2: (0, 0, 1, -1)  -> contributes to C10 and -C11
    # M3: (0, 1, 0, 1)   -> contributes to C01 and C11
    # M4: (1, 0, 1, 0)   -> contributes to C00 and C10
    # M5: (-1, 1, 0, 0)  -> contributes to -C00 and C01
    # M6: (0, 0, 0, 1)   -> contributes to C11
    # M7: (1, 0, 0, 0)   -> contributes to C00
    
    W_strassen = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0],   # M7
    ]
    
    # Now assign to global indices.
    # For block at columns (k, k+1):
    # U: same for all blocks (A is shared), indices 0-3
    # V: B indices are j*p + k' for j in {0,1}, k' in {k, k+1}
    #    So V[j*p + k + 0] for B[j][k], V[j*p + k + 1] for B[j][k+1]
    # W: C indices are k'*n + i for k' in {k, k+1}, i in {0,1}
    #    So W[(k+0)*n + i] for C[i][k], W[(k+1)*n + i] for C[i][k+1]
    
    col = 0
    for block in range(3):
        k = block * 2  # starting column of this block
        for r in range(7):
            # U
            for i in range(4):
                U[i][col] += U_strassen[r][i]
            # V: B[0][k], B[0][k+1], B[1][k], B[1][k+1]
            # Global V indices: 0*p+k, 0*p+k+1, 1*p+k, 1*p+k+1 = k, k+1, p+k, p+k+1
            V[k][col] += V_strassen[r][0]
            V[k + 1][col] += V_strassen[r][1]
            V[p + k][col] += V_strassen[r][2]
            V[p + k + 1][col] += V_strassen[r][3]
            # W: C[0][k], C[0][k+1], C[1][k], C[1][k+1]
            # Global W indices: k*n+0, k*n+1, (k+1)*n+0, (k+1)*n+1 = 2k, 2k+1, 2k+2, 2k+3
            W[2 * k][col] += W_strassen[r][0]
            W[2 * k + 1][col] += W_strassen[r][1]
            W[2 * k + 2][col] += W_strassen[r][2]
            W[2 * k + 3][col] += W_strassen[r][3]
            col += 1
    
    # Schoolbook for last column k=6
    # C[0][6] = A[0][0]*B[0][6] + A[0][1]*B[1][6]
    # C[1][6] = A[1][0]*B[0][6] + A[1][1]*B[1][6]
    for s in range(4):
        # A index: i*m + j, B index: j*p + k, C index: k*n + i
        if s == 0:
            i, j = 0, 0
        elif s == 1:
            i, j = 0, 1
        elif s == 2:
            i, j = 1, 0
        else:
            i, j = 1, 1
        U[i * m + j][col] = 1
        V[j * p + 6][col] = 1
        W[6 * n + i][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_2x2x7():
    """Schoolbook decomposition, rank 28."""
    n, m, p = 2, 2, 7
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

def construct():
    deadline = time.time() + 140  # safety margin
    
    # Try Strassen first
    try:
        result = _strassen_2x2x7()
        if _verify_decomposition(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        pass  # fallback below
    
    # Fallback: schoolbook
    result = _schoolbook_2x2x7()
    return result
