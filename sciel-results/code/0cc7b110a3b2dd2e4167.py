"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def _check_decomposition(U, V, W, tol=1e-10):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _build_strassen_block(k1, k2, U, V, W, col):
    """Build one Strassen block (rank 7) for column pair (k1, k2)."""
    n, m, p = DIMENSIONS
    
    # M1 = (a00 + a11) * (b00 + b11)
    U[0, col] = 1
    U[3, col] = 1
    V[0*p + k1, col] = 1
    V[1*p + k2, col] = 1
    W[k1*n + 0, col] = 1
    W[k2*n + 1, col] = 1
    col += 1
    
    # M2 = (a10 + a11) * b00
    U[2, col] = 1
    U[3, col] = 1
    V[0*p + k1, col] = 1
    W[k1*n + 1, col] = 1
    W[k2*n + 1, col] = -1
    col += 1
    
    # M3 = a00 * (b01 - b11)
    U[0, col] = 1
    V[0*p + k2, col] = 1
    V[1*p + k2, col] = -1
    W[k2*n + 0, col] = 1
    W[k2*n + 1, col] = 1
    col += 1
    
    # M4 = a11 * (b10 - b00)
    U[3, col] = 1
    V[1*p + k1, col] = 1
    V[0*p + k1, col] = -1
    W[k1*n + 0, col] = 1
    W[k1*n + 1, col] = 1
    col += 1
    
    # M5 = (a00 + a01) * b11
    U[0, col] = 1
    U[1, col] = 1
    V[1*p + k2, col] = 1
    W[k1*n + 0, col] = -1
    W[k2*n + 0, col] = 1
    col += 1
    
    # M6 = (a10 - a00) * (b00 + b01)
    U[2, col] = 1
    U[0, col] = -1
    V[0*p + k1, col] = 1
    V[0*p + k2, col] = 1
    W[k2*n + 1, col] = 1
    col += 1
    
    # M7 = (a01 - a11) * (b10 + b11)
    U[1, col] = 1
    U[3, col] = -1
    V[1*p + k1, col] = 1
    V[1*p + k2, col] = 1
    W[k1*n + 0, col] = 1
    col += 1
    
    return col

def _build_schoolbook_column(k, U, V, W, col):
    """Build one schoolbook column (rank 4)."""
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            U[i*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    return col

def construct():
    """Return a valid tensor decomposition for 2x2 * 2x13 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    # Use a slightly tighter deadline to ensure we return well before the 150s kill
    deadline = start_time + 145
    
    # Strategy: Use 6 Strassen blocks for first 12 columns, schoolbook for last column
    # Total rank: 6*7 + 4 = 46
    rank = 46
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for group in range(6):
        k1 = 2 * group
        k2 = 2 * group + 1
        col = _build_strassen_block(k1, k2, U, V, W, col)
    
    col = _build_schoolbook_column(12, U, V, W, col)
    
    # The construction is deterministic and fast, so this check should always pass
    # unless there is a bug in the block builders.
    if time.time() < deadline and _check_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: pure schoolbook (rank 52)
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m + j, c] = 1
                V_fb[j*p + k, c] = 1
                W_fb[k*n + i, c] = 1
                c += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
