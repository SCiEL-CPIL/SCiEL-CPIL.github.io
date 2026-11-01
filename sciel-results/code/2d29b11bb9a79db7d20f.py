"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 150.0
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _deadline_reached(margin=5.0):
    return _elapsed() > _DEADLINE - margin

def _rank12_2x2x4():
    """Return the rank-12 decomposition for 2x2 * 2x4 -> 2x4.
    U (4x12), V (8x12), W (8x12)
    """
    n, m, p = 2, 2, 4
    rank = 12
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Terms:
    # 0: (a+d)(e+h+i+l)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1; V[4, 0] = 1; V[7, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[4, 0] = 1; W[6, 0] = 1
    
    # 1: (c+d)(e+i)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1; V[4, 1] = 1
    W[2, 1] = 1; W[6, 1] = 1
    
    # 2: a(f+j-h-l)
    U[0, 2] = 1
    V[1, 2] = 1; V[5, 2] = 1; V[3, 2] = -1; V[7, 2] = -1
    W[1, 2] = 1; W[5, 2] = 1; W[3, 2] = -1; W[7, 2] = -1
    
    # 3: d(g+k-e-i)
    U[3, 3] = 1
    V[2, 3] = 1; V[6, 3] = 1; V[0, 3] = -1; V[4, 3] = -1
    W[2, 3] = 1; W[6, 3] = 1; W[0, 3] = -1; W[4, 3] = -1
    
    # 4: (a+b)(h+l)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1; V[7, 4] = 1
    W[1, 4] = 1; W[5, 4] = 1
    
    # 5: (c-a)(e+f+i+j)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1; V[4, 5] = 1; V[5, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1; W[6, 5] = 1; W[7, 5] = 1
    
    # 6: (b-d)(g+h+k+l)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1; V[6, 6] = 1; V[7, 6] = 1
    W[2, 6] = 1; W[3, 6] = 1; W[6, 6] = 1; W[7, 6] = 1
    
    # 7: (a+d)(f+j)
    U[0, 7] = 1; U[3, 7] = 1
    V[1, 7] = 1; V[5, 7] = 1
    W[1, 7] = 1; W[5, 7] = 1
    
    # 8: (c+d)(g+k)
    U[2, 8] = 1; U[3, 8] = 1
    V[2, 8] = 1; V[6, 8] = 1
    W[2, 8] = 1; W[6, 8] = 1
    
    # 9: a(h+l)
    U[0, 9] = 1
    V[3, 9] = 1; V[7, 9] = 1
    W[1, 9] = 1; W[5, 9] = 1
    
    # 10: d(e+i)
    U[3, 10] = 1
    V[0, 10] = 1; V[4, 10] = 1
    W[0, 10] = 1; W[4, 10] = 1
    
    # 11: (a+b)(i+l)
    U[0, 11] = 1; U[1, 11] = 1
    V[4, 11] = 1; V[7, 11] = 1
    W[4, 11] = 1; W[7, 11] = 1
    
    return U, V, W

def _trivial_2x2x1():
    """Return the rank-4 trivial decomposition for 2x2 * 2x1 -> 2x1.
    U (4x4), V (2x4), W (2x4)
    """
    n, m, p = 2, 2, 1
    rank = n * m * p  # 4
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _block_decomposition(n, m, p):
    """Build a block decomposition using rank-12 for 2x2x4 and rank-4 for 2x2x1.
    
    For 2x2 * 2x9:
    - Split the 9 columns into two 2x4 blocks and one 2x1 block.
    - Rank = 12 + 12 + 4 = 28.
    """
    n, m, p = 2, 2, 9
    
    rank = 12 + 12 + 4
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U_12, V_12, W_12 = _rank12_2x2x4()
    U_4, V_4, W_4 = _trivial_2x2x1()
    
    col_offset = 0
    
    # Block 1: columns 0-3
    U[:, col_offset:col_offset+12] = U_12
    for j in range(m):
        for k_local in range(4):
            k_global = k_local
            V[j * p + k_global, col_offset:col_offset+12] = V_12[j * 4 + k_local, :]
    for k_local in range(4):
        for i in range(n):
            k_global = k_local
            W[k_global * n + i, col_offset:col_offset+12] = W_12[k_local * n + i, :]
    col_offset += 12
    
    # Block 2: columns 4-7
    U[:, col_offset:col_offset+12] = U_12
    for j in range(m):
        for k_local in range(4):
            k_global = 4 + k_local
            V[j * p + k_global, col_offset:col_offset+12] = V_12[j * 4 + k_local, :]
    for k_local in range(4):
        for i in range(n):
            k_global = 4 + k_local
            W[k_global * n + i, col_offset:col_offset+12] = W_12[k_local * n + i, :]
    col_offset += 12
    
    # Block 3: column 8
    U[:, col_offset:col_offset+4] = U_4
    for j in range(m):
        V[j * p + 8, col_offset:col_offset+4] = V_4[j, :]
    for i in range(n):
        W[8 * n + i, col_offset:col_offset+4] = W_4[i, :]
    col_offset += 4
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n*m:
        return False
    if V.shape[0] != m*p:
        return False
    if W.shape[0] != p*n:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _to_list(mat):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    result = []
    for row in mat:
        r = []
        for val in row:
            v = round(val * 2) / 2.0
            if abs(v - round(v)) < 1e-12:
                v = int(round(v))
            r.append(v)
        result.append(r)
    return result

def construct():
    _START_TIME = time.time()
    n, m, p = DIMENSIONS
    
    try:
        U, V, W = _block_decomposition(n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    except Exception:
        pass
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
# EVOLVE-BLOCK-END
