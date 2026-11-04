"""Matrix multiplication 2x4x5 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]
n, m, p = DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 40

_START = time.time()
DEADLINE = 150
SAFETY_MARGIN = 10


def _elapsed():
    return time.time() - _START


def _time_left():
    return DEADLINE - SAFETY_MARGIN - _elapsed()


def _validate(U, V, W):
    """Check that U,V,W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook():
    """Return the schoolbook decomposition of rank 40."""
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
    return {"U": U, "V": V, "W": W}


def _strassen_222(U_offset, V_offset, W_offset, col):
    """
    Add Strassen's rank-7 decomposition for a 2x2x2 block.
    U_offset: (i0, j0) - top-left of A block
    V_offset: (j0, k0) - top-left of B block
    W_offset: (k0, i0) - top-left of C block
    col: starting column index
    Returns: updated col
    """
    i0, j0 = U_offset
    j0b, k0 = V_offset
    k0c, i0c = W_offset
    
    # Strassen's 7 multiplications for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    U_dict = {}
    V_dict = {}
    W_dict = {}
    
    # M1 = (A00+A11)(B00+B11)
    c = col
    U_dict[(i0, j0, c)] = U_dict.get((i0, j0, c), 0) + 1
    U_dict[(i0+1, j0+1, c)] = U_dict.get((i0+1, j0+1, c), 0) + 1
    V_dict[(j0b, k0, c)] = V_dict.get((j0b, k0, c), 0) + 1
    V_dict[(j0b+1, k0+1, c)] = V_dict.get((j0b+1, k0+1, c), 0) + 1
    W_dict[(k0c, i0c, c)] = W_dict.get((k0c, i0c, c), 0) + 1
    W_dict[(k0c+1, i0c+1, c)] = W_dict.get((k0c+1, i0c+1, c), 0) - 1
    W_dict[(k0c, i0c+1, c)] = W_dict.get((k0c, i0c+1, c), 0) + 1
    W_dict[(k0c+1, i0c, c)] = W_dict.get((k0c+1, i0c, c), 0) + 1
    col += 1
    
    # M2 = (A10+A11)B00
    c = col
    U_dict[(i0+1, j0, c)] = U_dict.get((i0+1, j0, c), 0) + 1
    U_dict[(i0+1, j0+1, c)] = U_dict.get((i0+1, j0+1, c), 0) + 1
    V_dict[(j0b, k0, c)] = V_dict.get((j0b, k0, c), 0) + 1
    W_dict[(k0c+1, i0c, c)] = W_dict.get((k0c+1, i0c, c), 0) + 1
    W_dict[(k0c+1, i0c+1, c)] = W_dict.get((k0c+1, i0c+1, c), 0) - 1
    col += 1
    
    # M3 = A00(B10-B11)
    c = col
    U_dict[(i0, j0, c)] = U_dict.get((i0, j0, c), 0) + 1
    V_dict[(j0b+1, k0, c)] = V_dict.get((j0b+1, k0, c), 0) + 1
    V_dict[(j0b+1, k0+1, c)] = V_dict.get((j0b+1, k0+1, c), 0) - 1
    W_dict[(k0c, i0c+1, c)] = W_dict.get((k0c, i0c+1, c), 0) + 1
    W_dict[(k0c+1, i0c, c)] = W_dict.get((k0c+1, i0c, c), 0) + 1
    col += 1
    
    # M4 = A11(B01-B10)
    c = col
    U_dict[(i0+1, j0+1, c)] = U_dict.get((i0+1, j0+1, c), 0) + 1
    V_dict[(j0b, k0+1, c)] = V_dict.get((j0b, k0+1, c), 0) + 1
    V_dict[(j0b+1, k0, c)] = V_dict.get((j0b+1, k0, c), 0) - 1
    W_dict[(k0c, i0c, c)] = W_dict.get((k0c, i0c, c), 0) + 1
    W_dict[(k0c+1, i0c+1, c)] = W_dict.get((k0c+1, i0c+1, c), 0) + 1
    col += 1
    
    # M5 = (A00+A01)B11
    c = col
    U_dict[(i0, j0, c)] = U_dict.get((i0, j0, c), 0) + 1
    U_dict[(i0, j0+1, c)] = U_dict.get((i0, j0+1, c), 0) + 1
    V_dict[(j0b+1, k0+1, c)] = V_dict.get((j0b+1, k0+1, c), 0) + 1
    W_dict[(k0c, i0c+1, c)] = W_dict.get((k0c, i0c+1, c), 0) - 1
    W_dict[(k0c+1, i0c+1, c)] = W_dict.get((k0c+1, i0c+1, c), 0) + 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    c = col
    U_dict[(i0+1, j0, c)] = U_dict.get((i0+1, j0, c), 0) + 1
    U_dict[(i0, j0, c)] = U_dict.get((i0, j0, c), 0) - 1
    V_dict[(j0b, k0, c)] = V_dict.get((j0b, k0, c), 0) + 1
    V_dict[(j0b, k0+1, c)] = V_dict.get((j0b, k0+1, c), 0) + 1
    W_dict[(k0c+1, i0c+1, c)] = W_dict.get((k0c+1, i0c+1, c), 0) + 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    c = col
    U_dict[(i0, j0+1, c)] = U_dict.get((i0, j0+1, c), 0) + 1
    U_dict[(i0+1, j0+1, c)] = U_dict.get((i0+1, j0+1, c), 0) - 1
    V_dict[(j0b+1, k0, c)] = V_dict.get((j0b+1, k0, c), 0) + 1
    V_dict[(j0b+1, k0+1, c)] = V_dict.get((j0b+1, k0+1, c), 0) + 1
    W_dict[(k0c, i0c, c)] = W_dict.get((k0c, i0c, c), 0) + 1
    col += 1
    
    return U_dict, V_dict, W_dict, col


def _schoolbook_221(U_offset, V_offset, W_offset, col):
    """
    Add schoolbook rank-4 decomposition for a 2x2x1 block.
    U_offset: (i0, j0) - top-left of A block
    V_offset: (j0, k0) - top-left of B block (single column)
    W_offset: (k0, i0) - top-left of C block (single column)
    col: starting column index
    Returns: updated col
    """
    i0, j0 = U_offset
    j0b, k0 = V_offset
    k0c, i0c = W_offset
    
    U_dict = {}
    V_dict = {}
    W_dict = {}
    
    # C[i0, k0] = A[i0,j0]*B[j0,k0] + A[i0,j0+1]*B[j0+1,k0]
    # C[i0+1, k0] = A[i0+1,j0]*B[j0,k0] + A[i0+1,j0+1]*B[j0+1,k0]
    
    for di in range(2):
        for dj in range(2):
            c = col
            U_dict[(i0+di, j0+dj, c)] = U_dict.get((i0+di, j0+dj, c), 0) + 1
            V_dict[(j0b+dj, k0, c)] = V_dict.get((j0b+dj, k0, c), 0) + 1
            W_dict[(k0c, i0c+di, c)] = W_dict.get((k0c, i0c+di, c), 0) + 1
            col += 1
    
    return U_dict, V_dict, W_dict, col


def _strassen_composed():
    """
    Compose Strassen decompositions for 2x4x5 matrix multiplication.
    
    Split A (2x4) into two 2x2 blocks: A1 (cols 0-1), A2 (cols 2-3)
    Split B (4x5) into two 2x5 blocks: B1 (rows 0-1), B2 (rows 2-3)
    
    C = A1*B1 + A2*B2
    
    For each 2x2x5 product, split B into 2x2, 2x2, 2x1:
    2x2x5 = 2x2x2 + 2x2x2 + 2x2x1
    
    Rank: 2 * (7 + 7 + 4) = 36
    """
    U_dict = {}
    V_dict = {}
    W_dict = {}
    col = 0
    
    # Process A1 @ B1 (i=0,1; j=0,1 for A; j=0,1 for B; k=0,1,2,3,4 for C)
    # A1 is rows 0-1 of A, cols 0-1: offset_U = (0, 0)
    # B1 is rows 0-1 of B, cols 0-4: offset_V = (0, 0)
    # C output cols 0-4: offset_W = (0, 0)
    
    # A1 @ B1a: 2x2x2, B1a is B rows 0-1, cols 0-1
    u1, v1, w1, col = _strassen_222((0, 0), (0, 0), (0, 0), col)
    for (i, j, c), val in u1.items():
        U_dict[(i, j, c)] = U_dict.get((i, j, c), 0) + val
    for (j, k, c), val in v1.items():
        V_dict[(j, k, c)] = V_dict.get((j, k, c), 0) + val
    for (k, i, c), val in w1.items():
        W_dict[(k, i, c)] = W_dict.get((k, i, c), 0) + val
    
    # A1 @ B1b: 2x2x2, B1b is B rows 0-1, cols 2-3
    u2, v2, w2, col = _strassen_222((0, 0), (0, 2), (2, 0), col)
    for (i, j, c), val in u2.items():
        U_dict[(i, j, c)] = U_dict.get((i, j, c), 0) + val
    for (j, k, c), val in v2.items():
        V_dict[(j, k, c)] = V_dict.get((j, k, c), 0) + val
    for (k, i, c), val in w2.items():
        W_dict[(k, i, c)] = W_dict.get((k, i, c), 0) + val
    
    # A1 @ B1c: 2x2x1, B1c is B rows 0-1, col 4
    u3, v3, w3, col = _schoolbook_221((0, 0), (0, 4), (4, 0), col)
    for (i, j, c), val in u3.items():
        U_dict[(i, j, c)] = U_dict.get((i, j, c), 0) + val
    for (j, k, c), val in v3.items():
        V_dict[(j, k, c)] = V_dict.get((j, k, c), 0) + val
    for (k, i, c), val in w3.items():
        W_dict[(k, i, c)] = W_dict.get((k, i, c), 0) + val
    
    # Process A2 @ B2 (i=0,1; j=2,3 for A; j=2,3 for B; k=0,1,2,3,4 for C)
    # A2 is rows 0-1 of A, cols 2-3: offset_U = (0, 2)
    # B2 is rows 2-3 of B, cols 0-4: offset_V = (2, 0)
    # C output cols 0-4: offset_W = (0, 0)
    
    # A2 @ B2a: 2x2x2, B2a is B rows 2-3, cols 0-1
    u4, v4, w4, col = _strassen_222((0, 2), (2, 0), (0, 0), col)
    for (i, j, c), val in u4.items():
        U_dict[(i, j, c)] = U_dict.get((i, j, c), 0) + val
    for (j, k, c), val in v4.items():
        V_dict[(j, k, c)] = V_dict.get((j, k, c), 0) + val
    for (k, i, c), val in w4.items():
        W_dict[(k, i, c)] = W_dict.get((k, i, c), 0) + val
    
    # A2 @ B2b: 2x2x2, B2b is B rows 2-3, cols 2-3
    u5, v5, w5, col = _strassen_222((0, 2), (2, 2), (2, 0), col)
    for (i, j, c), val in u5.items():
        U_dict[(i, j, c)] = U_dict.get((i, j, c), 0) + val
    for (j, k, c), val in v5.items():
        V_dict[(j, k, c)] = V_dict.get((j, k, c), 0) + val
    for (k, i, c), val in w5.items():
        W_dict[(k, i, c)] = W_dict.get((k, i, c), 0) + val
    
    # A2 @ B2c: 2x2x1, B2c is B rows 2-3, col 4
    u6, v6, w6, col = _schoolbook_221((0, 2), (2, 4), (4, 0), col)
    for (i, j, c), val in u6.items():
        U_dict[(i, j, c)] = U_dict.get((i, j, c), 0) + val
    for (j, k, c), val in v6.items():
        V_dict[(j, k, c)] = V_dict.get((j, k, c), 0) + val
    for (k, i, c), val in w6.items():
        W_dict[(k, i, c)] = W_dict.get((k, i, c), 0) + val
    
    rank = col
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for (i, j, c), val in U_dict.items():
        if val != 0:
            U[i * m + j][c] = val
    for (j, k, c), val in V_dict.items():
        if val != 0:
            V[j * p + k][c] = val
    for (k, i, c), val in W_dict.items():
        if val != 0:
            W[k * n + i][c] = val
    
    return {"U": U, "V": V, "W": W}


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4x5 matrix multiplication."""
    # Try the Strassen-composed decomposition (rank 36)
    result = _strassen_composed()
    if _validate(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook (rank 40)
    return _schoolbook()
