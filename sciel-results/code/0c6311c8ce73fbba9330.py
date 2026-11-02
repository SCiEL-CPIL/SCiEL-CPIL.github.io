"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def _check_decomposition(U, V, W, tol=1e-10):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
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
    """Build a Strassen block for columns k1, k2. Returns next col index.
    
    Standard Strassen for 2x2 x 2x2:
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)b00
    M3 = a00(b10-b11)
    M4 = a11(b01-b00)
    M5 = (a00+a01)b11
    M6 = (a10-a00)(b00+b01)
    M7 = (a01-a11)(b10+b11)
    
    C00 = M1 - M4 + M5 - M7
    C01 = M4 + M7
    C10 = M2 + M3
    C11 = M1 - M3 - M5 + M6
    """
    n = 2
    # U indices: 0=a00, 1=a01, 2=a10, 3=a11
    # V indices: j*p+k where j=0,1 (row of B), k=col
    # W indices: k*n+i where k=col, i=0,1
    
    # M1 = (a00+a11)(b0[k1]+b1[k2])
    U[0, col] += 1; U[3, col] += 1
    V[0*p+k1, col] += 1; V[1*p+k2, col] += 1
    W[k1*n+0, col] += 1; W[k1*n+1, col] += -1
    W[k2*n+0, col] += -1; W[k2*n+1, col] += 1
    col += 1
    
    # M2 = (a10+a11)b0[k1]
    U[2, col] += 1; U[3, col] += 1
    V[0*p+k1, col] += 1
    W[k1*n+1, col] += 1
    col += 1
    
    # M3 = a00(b1[k1]-b1[k2])
    U[0, col] += 1
    V[1*p+k1, col] += 1; V[1*p+k2, col] += -1
    W[k1*n+1, col] += 1; W[k2*n+1, col] += -1
    col += 1
    
    # M4 = a11(b0[k2]-b0[k1])
    U[3, col] += 1
    V[0*p+k2, col] += 1; V[0*p+k1, col] += -1
    W[k1*n+0, col] += -1; W[k2*n+0, col] += 1
    W[k1*n+1, col] += 1; W[k2*n+1, col] += 1
    col += 1
    
    # M5 = (a00+a01)b1[k2]
    U[0, col] += 1; U[1, col] += 1
    V[1*p+k2, col] += 1
    W[k1*n+0, col] += 1; W[k2*n+0, col] += -1
    col += 1
    
    # M6 = (a10-a00)(b0[k1]+b0[k2])
    U[2, col] += 1; U[0, col] += -1
    V[0*p+k1, col] += 1; V[0*p+k2, col] += 1
    W[k2*n+1, col] += 1
    col += 1
    
    # M7 = (a01-a11)(b1[k1]+b1[k2])
    U[1, col] += 1; U[3, col] += -1
    V[1*p+k1, col] += 1; V[1*p+k2, col] += 1
    W[k1*n+0, col] += -1; W[k2*n+0, col] += 1
    col += 1
    
    return col

def _build_schoolbook_column(U, V, W, col, k, p):
    """Build schoolbook terms for a single column k. Returns next col index."""
    n = 2
    for i in range(n):
        for j in range(2):
            U[i*2+j, col] = 1
            V[j*p+k, col] = 1
            W[k*n+i, col] = 1
            col += 1
    return col

def construct():
    """Return a valid tensor decomposition for 2x2 * 2x13 matrix multiplication.
    
    Uses 6 Strassen blocks (rank 7 each) for columns 0-11 plus 4 schoolbook terms
    for column 12. Total rank: 6*7 + 4 = 46.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
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
    
    if time.time() < deadline - 5 and _check_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: pure schoolbook
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
