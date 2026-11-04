import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 5]
_START_TIME = time.time()
_TIME_LIMIT = 130  # seconds, safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _validate_np(U, V, W, n, m, p):
    """
    Validate decomposition using NumPy.
    """
    R = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Compute the decomposition tensor D
    D = np.zeros((n*m, m*p, p*n), dtype=int)
    
    for r in range(R):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        D += np.einsum('i,j,k->ijk', u, v, w)
        
    # Construct Target Tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1
                
    if not np.array_equal(D, T):
        return False
    return True

def _get_strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    Shapes: (4, 7)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # P1: (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2: (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # P3: a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # P4: d(g-f)
    U[3, 3] = 1
    V[2, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # P5: (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # P6: (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # P7: (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _kron_strassen_4x4():
    """
    Returns U, V, W for 4x4x4 decomposition using Kronecker product of Strassen.
    Shapes: (16, 49)
    """
    U2, V2, W2 = _get_strassen_2x2()
    R2 = 7
    R4 = R2 * R2
    
    U4 = np.zeros((16, R4), dtype=int)
    V4 = np.zeros((16, R4), dtype=int)
    W4 = np.zeros((16, R4), dtype=int)
    
    for r1 in range(R2):
        for r2 in range(R2):
            r4 = r1 * R2 + r2
            for u1 in range(4):
                for u2 in range(4):
                    u = u1 * 2 + u2
                    U4[u, r4] = U2[u1, r1] * U2[u2, r2]
                    V4[u, r4] = V2[u1, r1] * V2[u2, r2]
                    W4[u, r4] = W2[u1, r1] * W2[u2, r2]
                    
    return U4, V4, W4

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u, r] = 1
                V[idx_v, r] = 1
                W[idx_w, r] = 1
                r += 1
                
    return U.tolist(), V.tolist(), W.tolist()

def _decompose_4x4x5():
    """
    Construct a rank-minimal decomposition for 4x4x5.
    Strategy: Use Kronecker product of Strassen (rank 49) for the 4x4x4 part,
    then extend to the 5th column with a rank-4 decomposition.
    Total rank: 49 + 4 = 53.
    """
    n, m, p = 4, 4, 5
    
    U4, V4, W4 = _kron_strassen_4x4()
    R4 = U4.shape[1]  # 49
    
    R_extra = 4
    R_new = R4 + R_extra
    
    U_new = np.zeros((n * m, R_new), dtype=int)
    V_new = np.zeros((m * p, R_new), dtype=int)
    W_new = np.zeros((p * n, R_new), dtype=int)
    
    # 1. Fill 4x4x4 part
    U_new[:, :R4] = U4
    
    # V indices: j*5+k (0..19).
    # V4 indices: j*4+k (0..15).
    # Map V4 row (j*4+k) to V_new row (j*5+k).
    for j in range(4):
        for k in range(4):
            v_idx_4x4 = j * 4 + k
            v_idx_4x5 = j * 5 + k
            V_new[v_idx_4x5, :R4] = V4[v_idx_4x4, :]
    
    # W indices: k*4+i (0..19).
    # W4 indices: k*4+i (0..15).
    # For k=0..3, indices match.
    W_new[:16, :R4] = W4
    
    # 2. Fill 4x4x1 part (k=4)
    # For r in 0..3:
    # U[i*4+j, r] = 1 if j==r
    # V[j*5+4, r] = 1 if j==r
    # W[4*4+i, r] = 1 for all i
    
    for r in range(R_extra):
        # U
        for i in range(4):
            for j in range(4):
                if j == r:
                    U_new[i * 4 + j, R4 + r] = 1
        
        # V
        for j in range(4):
            if j == r:
                V_new[j * 5 + 4, R4 + r] = 1
        
        # W
        for i in range(4):
            W_new[4 * 4 + i, R4 + r] = 1
    
    return U_new.tolist(), V_new.tolist(), W_new.tolist()

def construct():
    n, m, p = DIMENSIONS
    
    # Fallback to schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = (U_school, V_school, W_school)
    best_rank = n * m * p
    
    if n == 4 and m == 4 and p == 5:
        U_new, V_new, W_new = _decompose_4x4x5()
        R_new = len(U_new[0])
        
        if _validate_np(U_new, V_new, W_new, n, m, p):
            best_result = (U_new, V_new, W_new)
            best_rank = R_new
            
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
