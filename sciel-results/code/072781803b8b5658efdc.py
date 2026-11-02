# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0
_MARGIN = 10.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _MARGIN


def _verify(U, V, W, tol=1e-7):
    """Check exact decomposition: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all i,j,k."""
    n, m, p = 3, 3, 3
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook():
    n, m, p = 3, 3, 3
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _rank26_decomposition():
    """
    Construct a rank-26 decomposition for 3x3 matrix multiplication.
    Uses Strassen for the 2x2 top-left block and schoolbook for the rest.
    """
    n, m, p = 3, 3, 3
    nm, mp, pn = n*m, m*p, p*n
    rank = 26
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    # Strassen for 2x2 block (indices 0,1,3,4 in U,V,W)
    # U indices: 0(A00), 1(A01), 3(A10), 4(A11)
    # V indices: 0(B00), 1(B01), 3(B10), 4(B11)
    # W indices: 0(C00), 1(C01), 3(C10), 4(C11)
    
    strassen_terms = [
        # (U_vec, V_vec, W_vec)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, -1, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),   # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),  # M3
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 0, -1]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 1]),  # M5
        ([0, 1, 0, -1], [1, 1, 0, 0], [0, -1, 1, 0]), # M6
        ([0, 0, 1, -1], [0, 1, -1, 0], [1, 0, 0, 0]), # M7
    ]
    
    u_map = [0, 1, 3, 4]
    v_map = [0, 1, 3, 4]
    w_map = [0, 1, 3, 4]
    
    for u_s, v_s, w_s in strassen_terms:
        for idx, val in enumerate(u_s):
            if val != 0:
                U[u_map[idx], col] = val
        for idx, val in enumerate(v_s):
            if val != 0:
                V[v_map[idx], col] = val
        for idx, val in enumerate(w_s):
            if val != 0:
                W[w_map[idx], col] = val
        col += 1
        
    # Schoolbook for cross terms
    # a1*b2: A[0:2, 2] * B[2, 0:2]
    U[2, col] = 1.0; V[6, col] = 1.0; W[0, col] = 1.0; col += 1
    U[2, col] = 1.0; V[7, col] = 1.0; W[1, col] = 1.0; col += 1
    U[5, col] = 1.0; V[6, col] = 1.0; W[3, col] = 1.0; col += 1
    U[5, col] = 1.0; V[7, col] = 1.0; W[4, col] = 1.0; col += 1
    
    # A11*b1: A[0:2, 0:2] * B[0:2, 2]
    U[0, col] = 1.0; V[2, col] = 1.0; W[2, col] = 1.0; col += 1
    U[1, col] = 1.0; V[5, col] = 1.0; W[2, col] = 1.0; col += 1
    U[3, col] = 1.0; V[2, col] = 1.0; W[5, col] = 1.0; col += 1
    U[4, col] = 1.0; V[5, col] = 1.0; W[5, col] = 1.0; col += 1
    
    # a1*b3: A[0:2, 2] * B[2, 2]
    U[2, col] = 1.0; V[8, col] = 1.0; W[2, col] = 1.0; col += 1
    U[5, col] = 1.0; V[8, col] = 1.0; W[5, col] = 1.0; col += 1
    
    # a2*B11: A[2, 0:2] * B[0:2, 0:2]
    U[6, col] = 1.0; V[0, col] = 1.0; W[6, col] = 1.0; col += 1
    U[6, col] = 1.0; V[1, col] = 1.0; W[7, col] = 1.0; col += 1
    U[7, col] = 1.0; V[3, col] = 1.0; W[6, col] = 1.0; col += 1
    U[7, col] = 1.0; V[4, col] = 1.0; W[7, col] = 1.0; col += 1
    
    # a3*b2: A[2, 2] * B[2, 0:2]
    U[8, col] = 1.0; V[6, col] = 1.0; W[6, col] = 1.0; col += 1
    U[8, col] = 1.0; V[7, col] = 1.0; W[7, col] = 1.0; col += 1
    
    # a2*b1: A[2, 0:2] * B[0:2, 2]
    U[6, col] = 1.0; V[2, col] = 1.0; W[8, col] = 1.0; col += 1
    U[7, col] = 1.0; V[5, col] = 1.0; W[8, col] = 1.0; col += 1
    
    # a3*b3: A[2, 2] * B[2, 2]
    U[8, col] = 1.0; V[8, col] = 1.0; W[8, col] = 1.0; col += 1
    
    return U, V, W


def construct():
    n, m, p = 3, 3, 3
    U_s, V_s, W_s = _schoolbook()
    best = (U_s, V_s, W_s)
    
    if _time_left() > 5.0:
        U26, V26, W26 = _rank26_decomposition()
        if _verify(U26, V26, W26):
            best = (U26, V26, W26)
            
    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
