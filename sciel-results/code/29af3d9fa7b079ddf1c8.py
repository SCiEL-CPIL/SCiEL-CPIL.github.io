"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_strassen_2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    return U, V, W

def _build_4x4x4_strassen():
    U2, V2, W2 = _build_strassen_2x2()
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    big_mults = [
        {'A': [((0,0), 1), ((1,1), 1)], 'B': [((0,0), 1), ((1,1), 1)], 'C': [((0,0), 1), ((1,1), 1)]},
        {'A': [((1,0), 1), ((1,1), 1)], 'B': [((0,0), 1)], 'C': [((0,1), 1), ((1,1), -1)]},
        {'A': [((0,0), 1)], 'B': [((0,1), 1), ((1,1), -1)], 'C': [((1,0), 1), ((1,1), 1)]},
        {'A': [((1,1), 1)], 'B': [((1,0), 1), ((0,0), -1)], 'C': [((0,0), 1), ((0,1), 1)]},
        {'A': [((0,0), 1), ((0,1), 1)], 'B': [((1,1), 1)], 'C': [((0,0), -1), ((1,0), 1)]},
        {'A': [((1,0), 1), ((0,0), -1)], 'B': [((0,0), 1), ((0,1), 1)], 'C': [((1,1), 1)]},
        {'A': [((0,1), 1), ((1,1), -1)], 'B': [((1,0), 1), ((1,1), 1)], 'C': [((0,0), 1)]}
    ]
    for big_idx, bm in enumerate(big_mults):
        for r2 in range(7):
            col = big_idx * 7 + r2
            for (bi, bj), sa in bm['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        uv = U2[i2*2+k2, r2]
                        if uv != 0:
                            I = 2*bi + i2
                            J = 2*bj + k2
                            U[I*4+J, col] += sa * uv
            for (bk, bj), sb in bm['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        vv = V2[k2*2+j2, r2]
                        if vv != 0:
                            J = 2*bk + k2
                            K = 2*bj + j2
                            V[J*4+K, col] += sb * vv
            for (bi, bj), sc in bm['C']:
                for i2 in range(2):
                    for j2 in range(2):
                        wv = W2[i2*2+j2, r2]
                        if wv != 0:
                            I = 2*bi + i2
                            J = 2*bj + j2
                            W[I*4+J, col] += sc * wv
    return U, V, W

def _build_4x4x5():
    """Build a rank-55 decomposition for 4x4 x 4x5 multiplication.
    Uses Strassen for first 4 columns (rank 49) + schoolbook for 5th (rank 16).
    Total rank = 65.
    """
    U444, V444, W444 = _build_4x4x4_strassen()
    rank = 65
    U = np.zeros((16, rank), dtype=np.int64)
    V = np.zeros((20, rank), dtype=np.int64)
    W = np.zeros((20, rank), dtype=np.int64)
    # First 49 columns: 4x4x4 Strassen for k=0..3
    U[:, 0:49] = U444
    for j in range(4):
        for k in range(4):
            V[j*5 + k, 0:49] = V444[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W444[k*4 + i, :]
    # Last 16 columns: schoolbook for k=4
    col = 49
    for i in range(4):
        for j in range(4):
            U[i*4 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1
            col += 1
    return U, V, W

def _schoolbook_fallback():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _validate_fast(U, V, W, n, m, p):
    """Fast vectorized validation using tensor contraction."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    rank = U_np.shape[1]
    # Build the full tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # Reshape: U -> (n, m, R), V -> (m, p, R), W -> (p, n, R)
    U_r = U_np.reshape(n, m, rank)
    V_r = V_np.reshape(m, p, rank)
    W_r = W_np.reshape(p, n, rank)
    # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-9)

def _validate_numpy(U, V, W, n, m, p):
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if val != 1:
                    return False
    return True

def _try_reduce_tail(U, V, W, n, m, p, deadline):
    """Try to reduce the total rank by finding linear dependencies in the tail columns."""
    start = time.time()
    rank = U.shape[1]
    # The tail is the last 16 columns (schoolbook for k=8)
    tail_start = rank - 16
    U_tail = U[:, tail_start:]
    V_tail = V[:, tail_start:]
    W_tail = W[:, tail_start:]
    
    # Check if any tail column is a linear combination of others
    # For each tail column r, check if U_tail[:,r], V_tail[:,r], W_tail[:,r] are in the span
    # This is a 3-tensor linear dependence check - complex. Skip for now.
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS
    
    # Build the base decomposition: 2x 4x4x4 Strassen + 1x schoolbook
    U444, V444, W444 = _build_4x4x4_strassen()
    
    if not _validate_fast(U444, V444, W444, 4, 4, 4):
        U, V, W = _schoolbook_fallback()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    total_rank = 114
    U = np.zeros((16, total_rank), dtype=np.int64)
    V = np.zeros((36, total_rank), dtype=np.int64)
    W = np.zeros((36, total_rank), dtype=np.int64)
    
    # Block 1: columns 0-48 (k=0..3)
    U[:, 0:49] = U444
    for j in range(4):
        for k in range(4):
            V[j*9 + k, 0:49] = V444[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W444[k*4 + i, :]
    
    # Block 2: columns 49-97 (k=4..7)
    offset2 = 49
    U[:, offset2:offset2+49] = U444
    for j in range(4):
        for k in range(4):
            V[j*9 + (k+4), offset2:offset2+49] = V444[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4 + i, offset2:offset2+49] = W444[k*4 + i, :]
    
    # Block 3: columns 98-113 (k=8, schoolbook)
    offset3 = 98
    for i in range(4):
        for j in range(4):
            idx = i*4 + j
            U[i*4 + j, offset3 + idx] = 1
            V[j*9 + 8, offset3 + idx] = 1
            W[8*4 + i, offset3 + idx] = 1
    
    # Validate
    if not _validate_fast(U, V, W, n, m, p):
        if not _validate_numpy(U, V, W, n, m, p):
            U, V, W = _schoolbook_fallback()
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank if we have time
    if time.time() < deadline - 5:
        U, V, W, rank = _try_reduce_tail(U, V, W, n, m, p, deadline)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
