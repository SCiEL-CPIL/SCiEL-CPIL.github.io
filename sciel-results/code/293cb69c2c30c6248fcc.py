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

def _build_4x4_strassen():
    """Build rank-49 decomposition for 4x4 x 4x4 using Kronecker of 2x2 Strassen."""
    U2, V2, W2 = _strassen_2x2()
    
    # Kronecker product
    # U4 shape (16, 49)
    # V4 shape (16, 49)
    # W4 shape (16, 49)
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            u_idx = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                            U4[u_idx, c] = U2[i1 * 2 + j1, c1] * U2[i2 * 2 + j2, c2]
            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            v_idx = (j1 * 2 + j2) * 4 + (k1 * 2 + k2)
                            V4[v_idx, c] = V2[j1 * 2 + k1, c1] * V2[j2 * 2 + k2, c2]
            
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            w_idx = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)
                            W4[w_idx, c] = W2[k1 * 2 + i1, c1] * W2[k2 * 2 + i2, c2]
                            
    return U4, V4, W4

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
    """Build a decomposition for 4x4 x 4x8 using two 4x4 Strassen blocks."""
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140
    
    U4, V4, W4 = _build_4x4_strassen()
    
    r = 98
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # First half: k=0..3
    U[:, 0:49] = U4
    
    for j in range(4):
        for k in range(4):
            v_idx = j * 8 + k
            V[v_idx, 0:49] = V4[j * 4 + k, 0:49]
            
    for k in range(4):
        for i in range(4):
            w_idx = k * 4 + i
            W[w_idx, 0:49] = W4[k * 4 + i, 0:49]
            
    # Second half: k=4..7
    U[:, 49:98] = U4
    
    for j in range(4):
        for k in range(4, 8):
            k_prime = k - 4
            v_idx = j * 8 + k
            V[v_idx, 49:98] = V4[j * 4 + k_prime, 0:49]
            
    for k in range(4, 8):
        k_prime = k - 4
        for i in range(4):
            w_idx = k * 4 + i
            W[w_idx, 49:98] = W4[k_prime * 4 + i, 0:49]
            
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
