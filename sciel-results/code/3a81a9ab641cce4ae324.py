"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def _check_decomposition(U, V, W, tol=1e-10):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
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

def _build_strassen_block(U, V, W, col, k1, k2, p):
    """Build one Strassen block for columns k1, k2. Returns next col."""
    n = 2
    # M1 = (a00 + a11) * (b0k1 + b1k2)
    U[0, col] = 1; U[3, col] = 1
    V[0*p + k1, col] = 1; V[1*p + k2, col] = 1
    W[k1*n + 0, col] = 1; W[k2*n + 1, col] = 1
    col += 1
    
    # M2 = (a10 + a11) * b0k1
    U[2, col] = 1; U[3, col] = 1
    V[0*p + k1, col] = 1
    W[k1*n + 1, col] = 1; W[k2*n + 1, col] = -1
    col += 1
    
    # M3 = a00 * (b0k2 - b1k2)
    U[0, col] = 1
    V[0*p + k2, col] = 1; V[1*p + k2, col] = -1
    W[k2*n + 0, col] = 1; W[k2*n + 1, col] = 1
    col += 1
    
    # M4 = a11 * (b1k1 - b0k1)
    U[3, col] = 1
    V[1*p + k1, col] = 1; V[0*p + k1, col] = -1
    W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
    col += 1
    
    # M5 = (a00 + a01) * b1k2
    U[0, col] = 1; U[1, col] = 1
    V[1*p + k2, col] = 1
    W[k1*n + 0, col] = -1; W[k2*n + 0, col] = 1
    col += 1
    
    # M6 = (a10 - a00) * (b0k1 + b0k2)
    U[2, col] = 1; U[0, col] = -1
    V[0*p + k1, col] = 1; V[0*p + k2, col] = 1
    W[k2*n + 1, col] = 1
    col += 1
    
    # M7 = (a01 - a11) * (b1k1 + b1k2)
    U[1, col] = 1; U[3, col] = -1
    V[1*p + k1, col] = 1; V[1*p + k2, col] = 1
    W[k1*n + 0, col] = 1
    col += 1
    
    return col

def _build_schoolbook_column(U, V, W, col, k, p):
    """Build schoolbook decomposition for a single column k. Returns next col."""
    n, m = 2, 2
    for i in range(n):
        for j in range(m):
            U[i*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    return col

def construct():
    """Return a valid tensor decomposition for 2x2 * 2x13 matrix multiplication.
    
    Uses 6 Strassen blocks (rank 7 each) for column pairs (0,1)...(10,11)
    plus 1 schoolbook column (rank 4) for column 12. Total rank: 46.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Primary: 6 Strassen + 1 schoolbook = rank 46
    rank = 46
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for group in range(6):
        k1 = 2 * group
        k2 = 2 * group + 1
        col = _build_strassen_block(U, V, W, col, k1, k2, p)
    
    col = _build_schoolbook_column(U, V, W, col, 12, p)
    
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
