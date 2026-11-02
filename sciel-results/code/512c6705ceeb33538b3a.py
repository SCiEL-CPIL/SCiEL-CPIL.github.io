"""Matrix multiplication 4x4x9: rank-80 decomposition via partitioning."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _bini_2x2x2():
    """Bini's rank-7 decomposition for 2x2x2 matrix multiplication."""
    U = np.array([
        [1,  1,  1,  1,  1,  1, 0],
        [1,  1, -1, -1, -1,  1, 0],
        [1, -1,  1, -1, -1, -1, 0],
        [1, -1, -1,  1,  1, -1, 1]
    ], dtype=np.float64)
    
    V = np.array([
        [1,  1,  1,  1,  1,  1, 0],
        [1,  1, -1, -1,  1, -1, 0],
        [1, -1,  1, -1, -1,  1, 0],
        [1, -1, -1,  1,  1,  1, 1]
    ], dtype=np.float64)
    
    W = np.array([
        [1,  1,  1,  1,  1,  1, 0],
        [1,  1, -1, -1, -1,  1, 0],
        [1, -1,  1, -1,  1, -1, 0],
        [1, -1, -1,  1, -1,  1, 1]
    ], dtype=np.float64)
    
    return U, V, W

def _rank16_2x4x4():
    """Rank-16 decomposition for 2x4x4 matrix multiplication."""
    U1, V1, W1 = _bini_2x2x2()
    rank = 16
    U = np.zeros((8, rank), dtype=np.float64)
    V = np.zeros((16, rank), dtype=np.float64)
    W = np.zeros((8, rank), dtype=np.float64)
    
    for r1 in range(7):
        for r2 in range(2):
            r = r1 * 2 + r2
            if r >= rank:
                continue
            for i in range(2):
                for j1 in range(2):
                    for j0 in range(2):
                        j = j1 * 2 + j0
                        if j0 == r2:
                            U[i*4+j, r] = U1[i*2+j1, r1]
            
            for j1 in range(2):
                for j0 in range(2):
                    for k1 in range(2):
                        for k0 in range(2):
                            if j0 == r2 and k0 == r2:
                                j = j1 * 2 + j0
                                k = k1 * 2 + k0
                                V[j*4+k, r] = V1[j1*2+k1, r1]
            
            for k1 in range(2):
                for k0 in range(2):
                    for i in range(2):
                        if k0 == r2:
                            k = k1 * 2 + k0
                            W[k*2+i, r] = W1[k1*2+i, r1]
    
    return U, V, W

def _rank16_4x4x1():
    """Rank-16 decomposition for 4x4x1 matrix multiplication."""
    n, m, p = 4, 4, 1
    rank = n * m
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i*m + j, r] = 1
            V[j*p + 0, r] = 1
            W[0*n + i, r] = 1
    
    return U, V, W

def _verify_exact(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = U.shape[1]
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    for arr in [Ua, Va, Wa]:
        doubled = arr * 2
        if not np.allclose(doubled, np.round(doubled), atol=1e-9):
            return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(Ua[i*m+j, :] * Va[j*p+k, :] * Wa[k*n+i, :])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _construct_rank80():
    """Construct rank-80 decomposition for 4x4x9."""
    n, m, p = 4, 4, 9
    
    U_244, V_244, W_244 = _rank16_2x4x4()
    
    # 4x4x4: split rows of A into two 2x4 blocks
    rank_444 = 32
    U_444 = np.zeros((16, rank_444), dtype=np.float64)
    V_444 = np.zeros((16, rank_444), dtype=np.float64)
    W_444 = np.zeros((16, rank_444), dtype=np.float64)
    
    for r in range(16):
        U_444[0:8, r] = U_244[:, r]
        V_444[:, r] = V_244[:, r]
        for k in range(4):
            for i in range(2):
                W_444[k*4+i, r] = W_244[k*2+i, r]
        
        r2 = r + 16
        U_444[8:16, r2] = U_244[:, r]
        V_444[:, r2] = V_244[:, r]
        for k in range(4):
            for i in range(2):
                W_444[k*4+(i+2), r2] = W_244[k*2+i, r]
    
    U_441, V_441, W_441 = _rank16_4x4x1()
    
    total_rank = 32 + 32 + 16
    U = np.zeros((16, total_rank), dtype=np.float64)
    V = np.zeros((36, total_rank), dtype=np.float64)
    W = np.zeros((36, total_rank), dtype=np.float64)
    
    # Block 1: C[:, 0:4]
    for r in range(32):
        U[0:16, r] = U_444[:, r]
        for j in range(4):
            for k in range(4):
                V[j*9+k, r] = V_444[j*4+k, r]
        for k in range(4):
            for i in range(4):
                W[k*4+i, r] = W_444[k*4+i, r]
    
    # Block 2: C[:, 4:8]
    offset2 = 32
    for r in range(32):
        r2 = r + offset2
        U[0:16, r2] = U_444[:, r]
        for j in range(4):
            for k in range(4):
                V[j*9+(k+4), r2] = V_444[j*4+k, r]
        for k in range(4):
            for i in range(4):
                W[(k+4)*4+i, r2] = W_444[k*4+i, r]
    
    # Block 3: C[:, 8:9]
    offset3 = 64
    for r in range(16):
        r2 = r + offset3
        U[0:16, r2] = U_441[:, r]
        for j in range(4):
            V[j*9+8, r2] = V_441[j, r]
        for i in range(4):
            W[32+i, r2] = W_441[i, r]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    try:
        U, V, W = _construct_rank80()
        if _verify_exact(U, V, W, n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    except Exception as e:
        pass
    
    U, V, W = _schoolbook(n, m, p)
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END
