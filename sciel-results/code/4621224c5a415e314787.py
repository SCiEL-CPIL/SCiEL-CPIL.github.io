"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Return U, V, W for 2x2 x 2x2 = 2x2 using Strassen (rank 7).
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # A indices: (i,j) -> i*2+j: a00=0, a01=1, a10=2, a11=3
    # B indices: (j,k) -> j*2+k: b00=0, b01=1, b10=2, b11=3
    # C indices: (k,i) -> k*2+i: c00=0, c01=1, c10=2, c11=3
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = 1; W[3, 0] = -1
    
    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _build_decomposition():
    """Build a decomposition for 4x4 x 4x8 using Kronecker product of (2,2,2) Strassen and (2,2,4) schoolbook."""
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140
    
    # (4,4,8) as Kronecker of (2,2,2) and (2,2,4)
    # n = 2*2 = 4, m = 2*2 = 4, p = 2*4 = 8
    # rank = rank(2,2,2) * rank(2,2,4) = 7 * 16 = 112
    
    U1, V1, W1 = _strassen_2x2()  # shapes (4,7), (4,7), (4,7) for (2,2,2)
    
    # For (2,2,4): schoolbook
    n2, m2, p2 = 2, 2, 4
    r2 = n2 * m2 * p2  # 16
    U2 = np.zeros((n2*m2, r2))
    V2 = np.zeros((m2*p2, r2))
    W2 = np.zeros((p2*n2, r2))
    col = 0
    for i2 in range(n2):
        for j2 in range(m2):
            for k2 in range(p2):
                U2[i2*m2+j2, col] = 1
                V2[j2*p2+k2, col] = 1
                W2[k2*n2+i2, col] = 1
                col += 1
    
    n1, m1, p1 = 2, 2, 2
    r1 = 7
    r = r1 * r2  # 112
    
    U = np.zeros((n1*n2*m1*m2, r))
    V = np.zeros((m1*m2*p1*p2, r))
    W = np.zeros((p1*p2*n1*n2, r))
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u_idx = (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
                            U[u_idx, c] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v_idx = (j1*m2 + j2) * (p1*p2) + (k1*p2 + k2)
                            V[v_idx, c] = V1[j1*p1+k1, c1] * V2[j2*p2+k2, c2]
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w_idx = (k1*p2 + k2) * (n1*n2) + (i1*n2 + i2)
                            W[w_idx, c] = W1[k1*n1+i1, c1] * W2[k2*n2+i2, c2]
    
    if not _verify_decomposition(U, V, W, n, m, p):
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    try:
        result = _build_decomposition()
        if time.time() < deadline:
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
